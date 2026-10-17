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

from ToolsAgent.ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DESC

_cfg = Config()

# Read output types from the registry.

def _resolve_persist_dir() -> Path:
    try:
        configured = Path(str(_cfg.PERSIST_DIR)).expanduser()
        if not configured.is_absolute():
            configured = PROJECT_ROOT / configured
        return configured.resolve()
    except Exception as exc:
        raise RuntimeError(f"解析 KG 持久化目录失败: {exc}") from exc


def _resolve_bge_model_path() -> Path:
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
    staging_dir = None
    backup_dir = None
    try:
        persist_dir = _resolve_persist_dir()
        staging_dir = persist_dir.with_name(f"{persist_dir.name}.building")
        backup_dir = persist_dir.with_name(f"{persist_dir.name}.backup")

        print("=" * 60)
        print("  fix_my_index.py — 编程式确定性图谱构建")
        print("=" * 60)

        print("\n[1/4] 正在加载本地 BGE 向量模型...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=str(_resolve_bge_model_path())
        )

        # Use MockLLM for offline index construction.
        Settings.llm = MockLLM(max_tokens=1)

        # Remove stale temporary output, keeping the current index.
        if staging_dir.exists():
            print(f"清理上次未完成的临时目录: {staging_dir}")
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        triplets = []
        print("\n[2/4] 正在解析 tool_name_dict.py 并生成标准三元组...")
        for tool_name, info in BIOLOGY_TOOLS_DESC.items():
            if info.get("is_internal"):
                print(f"  [SKIP] internal tool: {tool_name}")
                continue

            print(f"  [+] {tool_name}")

            triplets.append((tool_name, "has the functionality that", info["description"]))

            triplets.append((tool_name, "is a", "Biology Tool"))

            input_types = info.get("input_types") or ["amino acid sequence"]
            for input_type in input_types:
                triplets.append((tool_name, "inputs", input_type))
                triplets.append((input_type, "is the input of", tool_name))

            # Read output types from the registry.
            output_types = info.get("output_types") or ["analysis result"]

            for output_type in output_types:
                triplets.append((tool_name, "outputs", output_type))
                triplets.append((output_type, "is the output of", tool_name))

            triplets.append((tool_name, "is sourced from", "Local computation"))

            triplets.append((tool_name, "does not need", "Security Check"))

        print(f"\n  [OK] total triplets: {len(triplets)}")

        print("\n[3/4] 正在构建知识图谱并计算向量...")
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        kg_index = KnowledgeGraphIndex([], storage_context=storage_context)

        embed_count = 0
        for triplet in triplets:
            include_embeddings = (triplet[1] == 'has the functionality that')
            kg_index.upsert_triplet(triplet, include_embeddings=include_embeddings)
            if include_embeddings:
                embed_count += 1
                print(f"  [EMB] ({triplet[0]}, {triplet[1]}, {triplet[2][:30]}...)")

        print("\n[4/4] 正在保存到磁盘...")
        kg_index.storage_context.persist(persist_dir=str(staging_dir))
        required_files = {"graph_store.json", "index_store.json", "docstore.json"}
        missing_files = [name for name in required_files if not (staging_dir / name).is_file()]
        if missing_files:
            raise RuntimeError(f"新索引缺少必要文件: {', '.join(missing_files)}")

        # Replace the index only after a successful build; restore it on failure.
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
