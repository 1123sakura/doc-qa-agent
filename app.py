# -*- coding: utf-8 -*-
"""
文档问答 Agent · Streamlit 网页界面（入口）

把命令行里的文档问答 Agent，变成一个能点开就用的网页。
在 PyCharm 里右键本文件 → Run 即可启动。

运行：python app.py   （或 streamlit run app.py）
"""
import os
# 自动从 .env 文件加载 DEEPSEEK_API_KEY 等环境变量
# 没装 python-dotenv 时也不报错，仍可用系统环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# 必须在 import agent 之前设置镜像，否则加载嵌入模型时会连 huggingface.co
# 默认走国内镜像（本地/国内可用）；部署到国外云时，在云平台环境变量里
# 设 HF_ENDPOINT=https://huggingface.co 覆盖即可（云端直连官方更快更稳）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from agent import init_agent, chat_once, load_memory, MEMORY_FILE


# ---------- 页面配置 ----------
st.set_page_config(page_title="文档问答 Agent", page_icon="📚", layout="centered")
st.title("📚 文档问答 Agent")
st.caption("基于你上传的文档，先检索再回答，答案带引用来源。")


# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("设置")

    # API Key：优先用环境变量，否则让用户在网页里输入
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        api_key = st.text_input("DeepSeek API Key", type="password", help="不填写则使用环境变量 DEEPSEEK_API_KEY")

    st.divider()
    st.markdown("""
    **项目说明**
    - 把 `.txt` / `.md` / `.pdf` 放进 `docs/`
    - 运行 `python rag.py` 建立索引
    - 然后在这里提问
    """)

    st.divider()
    st.subheader("🧠 长期记忆")
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


# ---------- 初始化 Agent（只执行一次） ----------
@st.cache_resource
def get_agent(api_key_value: str):
    return init_agent(api_key=api_key_value or None)


# 启动时确保索引存在（云端首次部署会自动构建；本地可提前用 python rag.py 建好）
if not os.path.exists("faiss_index"):
    from rag import build_index   # 仅首次建索引时才加载 rag（含 torch）
    with st.spinner("首次构建向量索引中（约 1-2 分钟，需要下载嵌入模型）..."):
        build_index()

# 没有 Key 时先不要初始化 Agent（否则会卡在终端等待输入），
# 提示用户去侧边栏填写，填完页面会自动刷新并加载。
if not api_key:
    st.info("👈 请在左侧侧边栏输入你的 DeepSeek API Key，输入后页面会自动刷新并加载 Agent。")
    st.stop()

try:
    with st.spinner("正在加载模型与索引（首次约 30 秒，请稍候）..."):
        graph = get_agent(api_key)
except Exception as e:
    st.error(f"启动 Agent 失败：{e}")
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
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 调用 Agent（显示加载动画 + 错误处理，避免白屏）
    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索文档并思考中..."):
                # 把 streamlit 里的历史转成 LangChain 消息对象
                history = []
                for m in st.session_state.messages[:-1]:
                    if m["role"] == "user":
                        history.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        history.append(AIMessage(content=m["content"]))

                reply = chat_once(graph, history, user_input)

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(
                f"⚠️ 调用失败：{e}\n\n"
                "常见原因：API Key 无效 / 网络不通 / 额度不足。请检查后重试。"
            )
