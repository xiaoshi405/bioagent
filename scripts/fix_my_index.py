import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "LLAMA_INDEX_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "llama_index")
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from llama_index.core import StorageContext, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.llms import MockLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from app.core.config import Config

# 导入工具字典
from ToolsAgent.ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DESC

# 统一配置读取
_cfg = Config()

# 修改记录：
# - 2026-08-31：KG 构建改用本地 MockLLM，不再读取 API Key 或初始化
#   云端客户端，保证离线构建不会外发工具注册表与类型拓扑。
# - 2026-07-19：从工具注册表的 output_types 字段读取明确输出类型，
#   避免新增 Biology 工具在 KG 中退化为通用 analysis result。
# - 2026-07-19：改为临时目录构建并在成功后替换旧索引，避免构建失败
#   导致当前可用 storage 被提前删除。
# - 2026-07-20：从注册表读取 input_types，使PDB文件与氨基酸序列在KG中
#   成为不同输入类型，并支持PDB到序列的工具连接。

def _resolve_persist_dir() -> Path:
    """把配置中的索引路径稳定解析为绝对路径。"""
    try:
        configured = Path(str(_cfg.PERSIST_DIR)).expanduser()
        if not configured.is_absolute():
            configured = PROJECT_ROOT / configured
        return configured.resolve()
    except Exception as exc:
        raise RuntimeError(f"解析 KG 持久化目录失败: {exc}") from exc


def _resolve_bge_model_path() -> Path:
    """把本地 BGE 配置解析到项目目录，并禁止意外回退到网络下载。"""
    try:
        configured = Path(str(_cfg.BGE_MODEL_PATH)).expanduser()
        if not configured.is_absolute():
            configured = PROJECT_ROOT / configured
        resolved = configured.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"本地 BGE 模型目录不存在: {resolved}")
        return resolved
    except Exception as exc:
        raise RuntimeError(f"解析本地 BGE 模型目录失败: {exc}") from exc


def refresh_biology_index():
    """离线重建 Biology KG；仅在新索引完整落盘后替换旧索引。"""
    staging_dir = None
    backup_dir = None
    try:
        persist_dir = _resolve_persist_dir()
        staging_dir = persist_dir.with_name(f"{persist_dir.name}.building")
        backup_dir = persist_dir.with_name(f"{persist_dir.name}.backup")

        print("=" * 60)
        print("  fix_my_index.py — 编程式确定性图谱构建")
        print("=" * 60)

        # 1. 配置本地 Embedding（BGE 模型，完全离线，不消耗 API）
        print("\n[1/4] 正在加载本地 BGE 向量模型...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=str(_resolve_bge_model_path())
        )

        # 2. 只用本地 MockLLM 满足 LlamaIndex 初始化，不读取密钥或访问网络。
        Settings.llm = MockLLM(max_tokens=1)

        # 3. 只清理上次失败遗留的临时目录，绝不提前删除当前索引。
        if staging_dir.exists():
            print(f"清理上次未完成的临时目录: {staging_dir}")
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        # 4. 从唯一工具注册表组装兼容 custom_kg_retrievers 的三元组。
        triplets = []
        print("\n[2/4] 正在解析 tool_name_dict.py 并生成标准三元组...")
        for tool_name, info in BIOLOGY_TOOLS_DESC.items():
            if info.get("is_internal"):
                print(f"  [SKIP] internal tool: {tool_name}")
                continue

            print(f"  [+] {tool_name}")

            # 关系 1: has the functionality that (用于检索器相似度匹配)
            triplets.append((tool_name, "has the functionality that", info["description"]))

            # 关系 2: is a
            triplets.append((tool_name, "is a", "Biology Tool"))

            # 关系 3: inputs & is the input of (BFS 寻路必需)
            input_types = info.get("input_types") or ["amino acid sequence"]
            for input_type in input_types:
                triplets.append((tool_name, "inputs", input_type))
                triplets.append((input_type, "is the input of", tool_name))

            # 关系 4: outputs & is the output of (BFS 寻路必需)
            # 明确输出类型由唯一工具注册表维护，避免脚本用名称猜测工具语义。
            output_types = info.get("output_types") or ["analysis result"]

            for output_type in output_types:
                triplets.append((tool_name, "outputs", output_type))
                triplets.append((output_type, "is the output of", tool_name))

            # 关系 5: is sourced from
            triplets.append((tool_name, "is sourced from", "Local computation"))

            # 关系 6: 安全验证
            triplets.append((tool_name, "does not need", "Security Check"))

        print(f"\n  [OK] total triplets: {len(triplets)}")

        # 5. 构建空图谱并手动插入组装好的三元组
        print("\n[3/4] 正在构建知识图谱并计算向量...")
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        kg_index = KnowledgeGraphIndex([], storage_context=storage_context)

        embed_count = 0
        for triplet in triplets:
            # 只对含有功能描述的关系计算 Embedding
            include_embeddings = (triplet[1] == 'has the functionality that')
            kg_index.upsert_triplet(triplet, include_embeddings=include_embeddings)
            if include_embeddings:
                embed_count += 1
                print(f"  [EMB] ({triplet[0]}, {triplet[1]}, {triplet[2][:30]}...)")

        # 6. 持久化到本地
        print("\n[4/4] 正在保存到磁盘...")
        kg_index.storage_context.persist(persist_dir=str(staging_dir))
        required_files = {"graph_store.json", "index_store.json", "docstore.json"}
        missing_files = [name for name in required_files if not (staging_dir / name).is_file()]
        if missing_files:
            raise RuntimeError(f"新索引缺少必要文件: {', '.join(missing_files)}")

        # 新索引完整后再切换；切换异常时自动恢复旧索引。
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if persist_dir.exists():
            persist_dir.rename(backup_dir)
        try:
            staging_dir.rename(persist_dir)
        except Exception:
            if backup_dir.exists() and not persist_dir.exists():
                backup_dir.rename(persist_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        print("\n" + "=" * 60)
        print("  [DONE] Graph built successfully!")
        print(f"  Total triplets: {len(triplets)}")
        print(f"  Triplets with embeddings: {embed_count}")
        print(f"  Saved to: {persist_dir}")
        print("=" * 60)
        return {
            "status": "success",
            "persist_dir": str(persist_dir),
            "triplet_count": len(triplets),
            "embedding_count": embed_count,
        }
    except Exception as exc:
        print(f"[ERROR] KG 索引构建失败，旧索引保持不变: {exc}")
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        return {"status": "error", "message": str(exc)}


if __name__ == "__main__":
    result = refresh_biology_index()
    if result.get("status") != "success":
        sys.exit(1)
