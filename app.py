# -*- coding: utf-8 -*-
"""
文档问答 Agent · Streamlit 网页界面（入口）

修复启动黑屏的核心思路：
- 顶部不再 import agent，避免一次性拖入 torch/transformers。
- 先渲染页面骨架（标题、侧边栏、输入框），让用户立刻看到界面。
- 用户第一次输入问题时，再在 spinner 里懒加载 Agent 核心与模型。
"""
import os

# 自动从 .env 文件加载 DASHSCOPE_API_KEY 等环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# ---------- 页面骨架（先显示，不 import 任何重依赖）----------
st.set_page_config(page_title="文档问答 Agent", page_icon="📚", layout="centered")
st.title("📚 文档问答 Agent")
st.caption("基于你上传的文档，先检索再回答，答案带引用来源。")


# ---------- 缓存初始化 Agent（避免重复加载，同时避免启动黑屏）----------
@st.cache_resource(show_spinner=False)
def _init_graph_cached(key: str):
    from agent import init_agent
    return init_agent(key)


# ---------- 侧边栏（只放设置和说明，记忆等 agent 加载后再显示）----------
api_key = os.environ.get("DASHSCOPE_API_KEY", "")

# 调试日志：确认环境变量是否正确加载（不打印完整 Key）
if api_key:
    print(f"[DEBUG] DASHSCOPE_API_KEY loaded: length={len(api_key)}, prefix={api_key[:10]}..., suffix=...{api_key[-5:]}")
else:
    print("[DEBUG] DASHSCOPE_API_KEY NOT loaded from environment")

with st.sidebar:
    st.header("设置")

    if not api_key:
        api_key = st.text_input(
            "阿里云 DashScope API Key",
            type="password",
            help="不填写则使用环境变量 DASHSCOPE_API_KEY",
        )

    st.divider()
    st.markdown("""
    **项目说明**
    - 把 `.txt` / `.md` / `.pdf` 放进 `docs/`
    - 首次提问时会调用 DashScope Embedding 构建/加载索引
    - 回答基于 `docs/` 里的文档内容
    """)


# ---------- 没有 Key 时直接停掉，不加载 Agent ----------
if not api_key:
    st.info("👈 请在左侧侧边栏输入你的阿里云 DashScope API Key，输入后即可提问。")
    st.stop()


# ---------- 对话历史 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- 用户输入 ----------
user_input = st.chat_input("请输入你的问题，例如：年假怎么请？")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()


# ---------- 如果最后一条是用户消息，则生成回复 ----------
needs_reply = (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
)

if needs_reply:
    user_input = st.session_state.messages[-1]["content"]

    # 先确保 graph 已初始化（不在 chat_message 内部做 import，避免 DOM 同步问题）
    if "graph" not in st.session_state:
        with st.spinner("首次启动，正在初始化 Agent（约 10-20 秒）..."):
            st.session_state.graph = _init_graph_cached(api_key)

    graph = st.session_state.graph
    from langchain_core.messages import HumanMessage, AIMessage
    from agent import chat_once

    try:
        with st.chat_message("assistant"):
            with st.spinner("正在检索文档并思考中..."):
                history = []
                for m in st.session_state.messages[:-1]:
                    if m["role"] == "user":
                        history.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        history.append(AIMessage(content=m["content"]))

                reply = chat_once(graph, history, user_input)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()
    except Exception as e:
        st.error(
            f"⚠️ 调用失败：{e}\n\n"
            "常见原因：API Key 无效 / 网络不通 / 额度不足。请检查后重试。"
        )


# ---------- 长期记忆 ----------
if "graph" in st.session_state:
    with st.sidebar:
        st.divider()
        st.subheader("🧠 长期记忆")
        from agent import load_memory, MEMORY_FILE
        mem_entries = load_memory()
        if mem_entries:
            for e in mem_entries:
                st.caption(f"• {e}")
        else:
            st.caption("（暂无记忆）")
        if st.button("🗑 清空长期记忆"):
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
            st.rerun()
