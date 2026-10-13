# -*- coding: utf-8 -*-
"""
BioAgent Web Demo — 带推理链可视化的生命科学智能助手前端。

修改记录：
- 2026-07-20：增加受控PDB文件上传，把结构文件路径传入Agent工具链。
- 2026-07-24：增加受控单记录Protein FASTA上传。
- 2026-07-26：展示v0.3b脱敏类型规划轨迹、修复后计划和Artifact流转。
"""
import hashlib
import re
from pathlib import Path

import streamlit as st
import requests
from app.core.config import Config

_cfg = Config()


def save_uploaded_pdb(uploaded_file):
    """把前端PDB保存到受控目录，返回绝对路径或错误信息。"""
    try:
        if uploaded_file is None:
            return None, None
        original_name = Path(str(uploaded_file.name)).name
        if Path(original_name).suffix.lower() != ".pdb":
            return None, "仅支持上传 .pdb 文件"

        content = uploaded_file.getvalue()
        max_bytes = _cfg.MAX_PDB_UPLOAD_MB * 1024 * 1024
        if not content:
            return None, "上传的PDB文件为空"
        if len(content) > max_bytes:
            return None, f"PDB文件不能超过 {_cfg.MAX_PDB_UPLOAD_MB} MB"

        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem)
        safe_stem = safe_stem.strip("._") or "uploaded_structure"
        digest = hashlib.sha256(content).hexdigest()[:12]
        upload_dir = Path(_cfg.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        destination = upload_dir / f"{safe_stem}_{digest}.pdb"
        if not destination.exists():
            destination.write_bytes(content)
        return str(destination.resolve()), None
    except Exception as exc:
        return None, f"保存PDB文件失败: {exc}"


def save_uploaded_fasta(uploaded_file):
    """把前端Protein FASTA保存到受控目录，返回绝对路径或错误信息。"""
    try:
        if uploaded_file is None:
            return None, None
        original_name = Path(str(uploaded_file.name)).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in {".fasta", ".fa", ".faa"}:
            return None, "仅支持上传 .fasta、.fa 或 .faa 文件"

        content = uploaded_file.getvalue()
        max_bytes = _cfg.MAX_FASTA_UPLOAD_MB * 1024 * 1024
        if not content:
            return None, "上传的FASTA文件为空"
        if len(content) > max_bytes:
            return None, f"FASTA文件不能超过 {_cfg.MAX_FASTA_UPLOAD_MB} MB"

        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem)
        safe_stem = safe_stem.strip("._") or "uploaded_protein"
        digest = hashlib.sha256(content).hexdigest()[:12]
        upload_dir = Path(_cfg.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        destination = upload_dir / f"{safe_stem}_{digest}{suffix}"
        if not destination.exists():
            destination.write_bytes(content)
        return str(destination.resolve()), None
    except Exception as exc:
        return None, f"保存FASTA文件失败: {exc}"


# ─────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BioAgent · 蛋白质工具智能助手",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 全局 CSS：渐变标题 + 消息气泡 + 卡片样式
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* 隐藏默认 header */
    #MainMenu, footer {visibility: hidden;}

    /* 渐变大标题 */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3DD6AC 0%, #5B8DEF 60%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* 推理步骤徽章 */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .badge-blue   { background: #1E3A5F; color: #60A5FA; border: 1px solid #2563EB; }
    .badge-green  { background: #14532D; color: #34D399; border: 1px solid #059669; }
    .badge-purple { background: #3B1F6B; color: #C084FC; border: 1px solid #7C3AED; }

    /* 步骤卡片 */
    .step-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 130, 201, 0.25);
        border-left: 3px solid #5B8DEF;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .step-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #93C5FD;
        margin-bottom: 4px;
    }
    .step-body {
        font-size: 0.82rem;
        color: #CBD5E1;
        font-family: monospace;
        white-space: pre-wrap;
        word-break: break-all;
    }

    /* 时延标签 */
    .latency-tag {
        font-size: 0.75rem;
        color: #64748B;
        padding: 2px 8px;
        border: 1px solid #334155;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 侧边栏
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 服务配置")
    st.caption("模型、API 地址和密钥由后端 .env 配置，修改后请重启 Agent 服务。")
    backend_url = st.text_input("后端地址", value=f"http://127.0.0.1:{_cfg.SERVER_PORT}")

    st.divider()
    st.markdown("### 🧩 PDB结构文件")
    uploaded_pdb = st.file_uploader(
        "上传本地PDB文件",
        type=["pdb"],
        accept_multiple_files=False,
        help=f"文件只保存在本地项目uploads目录，大小上限{_cfg.MAX_PDB_UPLOAD_MB} MB",
    )
    if uploaded_pdb is not None:
        st.caption(f"已选择：{Path(uploaded_pdb.name).name}")

    st.markdown("### 🧬 Protein FASTA文件")
    uploaded_fasta = st.file_uploader(
        "上传单记录Protein FASTA",
        type=["fasta", "fa", "faa"],
        accept_multiple_files=False,
        help=(
            "仅支持单条蛋白质记录，文件保存在本地uploads目录，"
            f"大小上限{_cfg.MAX_FASTA_UPLOAD_MB} MB"
        ),
    )
    if uploaded_fasta is not None:
        st.caption(f"已选择：{Path(uploaded_fasta.name).name}")
    
    st.divider()
    st.markdown("### 📖 示例问题")
    examples = [
        "计算序列 MALWMRLLPLLALLALWGPDPAAA 的等电点和分子量",
        "这段序列有信号肽吗？MKTLLLTLVVVTIVCLDLGAVGNGSSLSGSGPASTCKSYDMTFNPFNYQAYISDSGSSVTCINNLNTTAQTASTFRRIASGSGAYVSDTCMPFQNCLSPNFGCRSNCPAGKIVKLDNKTWPKAQYNIVAKSTLPLKEFADMK",
        "分析蛋白质 MSSANTTDEAHLQGCTSAGRSSGSSTSSRPRSNMGSSGSGR 的跨膜区和溶解度",
        "对序列 ACDEFGHIKLMNPQRSTVWY 做一站式全分析",
        "分析我上传的PDB结构，报告链、残基、原子和二级结构注释",
    ]
    for ex in examples:
        if st.button(f"💬 {ex[:38]}...", key=ex, use_container_width=True):
            st.session_state["pending_input"] = ex

    st.divider()
    st.markdown(f"**模型**: `{_cfg.MODEL_NAME}`")
    st.markdown(f"**Embedding**: BGE-Small-zh")
    st.caption("启动命令：python scripts/start_demo.py")

# ─────────────────────────────────────────────
# 主区域：标题
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🧬 BioAgent</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">知识图谱驱动的生命科学多模态智能体 · 自动规划工具链 · 多步串联推理</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 会话初始化
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


# ─────────────────────────────────────────────
# 辅助：渲染推理链面板
# ─────────────────────────────────────────────
def render_reasoning(reasoning: dict, latency: float = None):
    retrieved = reasoning.get("retrieved_tools", [])
    plan_chain = reasoning.get("plan_chain", [])
    executed_chain = reasoning.get("plan_chain_executed", [])
    tool_steps = reasoning.get("tool_steps", [])
    typed_planning = reasoning.get("typed_planning") or {}

    header = "🔍 Agent 推理链"
    if latency:
        header += f"  <span class='latency-tag'>⏱ {latency}s</span>"

    with st.expander(header, expanded=False):
        # --- 检索到的工具 ---
        st.markdown("**① 知识图谱检索结果**")
        if retrieved:
            badges_html = "".join(
                f"<span class='badge badge-blue'>🔧 {t}</span>" for t in retrieved
            )
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.caption("暂无检索结果")

        st.markdown("---")

        # --- Plan Chain ---
        st.markdown("**② LLM 规划的工具调用链（Plan Chain）**")
        if plan_chain:
            chain_html = " → ".join(
                f"<span class='badge badge-purple'>⚙️ {t}</span>" for t in plan_chain
            )
            st.markdown(chain_html, unsafe_allow_html=True)
            if executed_chain and executed_chain != plan_chain:
                st.caption("类型检查/修复后的实际执行链")
                executed_html = " → ".join(
                    f"<span class='badge badge-green'>✅ {t}</span>"
                    for t in executed_chain
                )
                st.markdown(executed_html, unsafe_allow_html=True)
        else:
            st.caption("暂无规划")

        st.markdown("---")

        # --- 工具执行详情 ---
        st.markdown("**③ 工具逐步执行详情**")
        if tool_steps:
            for idx, step in enumerate(tool_steps, 1):
                tool_name = step.get("tool_name", "unknown")
                if typed_planning:
                    input_ids = step.get("input_artifact_ids") or {}
                    output_ids = step.get("output_artifact_ids") or []
                    duration_ms = step.get("duration_ms")
                    repair_text = (
                        " · 自动修复插入"
                        if step.get("inserted_by_repair")
                        else ""
                    )
                    st.markdown(
                        f"**Step {idx} · {tool_name}{repair_text}**"
                    )
                    st.caption(
                        f"输入Artifact：{input_ids or '无'} ｜ "
                        f"输出Artifact：{output_ids or '无'} ｜ "
                        f"耗时：{duration_ms if duration_ms is not None else '未知'} ms"
                    )
                else:
                    args_prev = step.get("args_preview", "")
                    out_prev = step.get("output_preview", "")

                    st.markdown(f"""
                    <div class="step-card">
                        <div class="step-title">Step {idx} · <span class='badge badge-green'>{tool_name}</span></div>
                        <div class="step-body">📥 输入预览: {args_prev[:120]}</div>
                        <div class="step-body" style="margin-top:4px">📤 输出预览: {out_prev[:300]}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.caption("暂无工具执行记录")

        if typed_planning:
            st.markdown("---")
            st.markdown("**④ 类型约束与Artifact轨迹（v0.3b）**")
            initial_items = typed_planning.get("initial_artifacts") or []
            initial_types = [
                item.get("type_name", "unknown")
                for item in initial_items
                if isinstance(item, dict)
            ]
            st.caption(
                f"状态：{typed_planning.get('status', 'unknown')} ｜ "
                f"初始类型：{initial_types or '无'} ｜ "
                f"尝试修复：{typed_planning.get('repair_attempted', False)} ｜ "
                f"应用修复：{typed_planning.get('repair_applied', False)}"
            )
            artifact_pool = typed_planning.get("artifact_pool") or {}
            if artifact_pool:
                st.caption(
                    f"对象池Artifact数量："
                    f"{artifact_pool.get('artifact_count', 0)}"
                )
            planning_issues = typed_planning.get("planning_issues") or []
            if planning_issues:
                st.warning(
                    "；".join(
                        str(item.get("message", item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in planning_issues
                    )
                )
            if typed_planning.get("error_type"):
                st.error(
                    f"阻断类型：{typed_planning.get('error_type')}"
                )


# ─────────────────────────────────────────────
# 历史消息渲染
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "reasoning" in msg:
            render_reasoning(msg["reasoning"], msg.get("latency"))


# ─────────────────────────────────────────────
# 用户输入（支持侧边栏示例按钮填入）
# ─────────────────────────────────────────────
pending = st.session_state.pop("pending_input", None)
prompt = st.chat_input("输入生物学问题，例如：计算序列 MASKL... 的等电点") or pending

if prompt:
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用后端
    with st.chat_message("assistant"):
        with st.spinner("🧠 Agent 正在检索知识图谱并调用生物工具..."):
            try:
                pdb_path, pdb_error = save_uploaded_pdb(uploaded_pdb)
                if pdb_error:
                    raise ValueError(pdb_error)
                fasta_path, fasta_error = save_uploaded_fasta(uploaded_fasta)
                if fasta_error:
                    raise ValueError(fasta_error)
                file_path_list = [
                    path for path in [pdb_path, fasta_path] if path
                ]
                payload = {
                    "question": prompt,
                    "is_add_filename": bool(file_path_list),
                    "file_path_list": file_path_list
                }
                resp = requests.post(
                    f"{backend_url}/chat",
                    json=payload,
                    timeout=180
                )

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("error_type"):
                        raise RuntimeError(f"{data['error_type']}: {data.get('message', '后端请求失败')}")
                    ans = data.get("response", "未获取到结果")
                    reasoning = data.get("reasoning", {})
                    latency = data.get("latency", None)

                    # 渲染答案
                    st.markdown(ans)

                    # 渲染推理链
                    render_reasoning(reasoning, latency)

                    # 存入历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans,
                        "reasoning": reasoning,
                        "latency": latency,
                    })
                else:
                    err = f"❌ 后端响应失败，状态码：{resp.status_code}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

            except requests.exceptions.ConnectionError:
                err = "❌ 无法连接后端服务，请运行 `python scripts/start_demo.py`，或按安装指南分别启动服务"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"❌ 发生异常: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
