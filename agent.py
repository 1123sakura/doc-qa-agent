# -*- coding: utf-8 -*-
"""
文档问答 Agent · LangGraph 编排

把 rag.py 里的 retrieve_docs 检索工具，挂进 LangGraph Agent。
结果：一个"会先查文档、再回答、并带引用来源"的 Agent。

运行：python agent.py   # 命令行模式，直接问答
"""

import os
import getpass

# 必须在任何可能引入 huggingface_hub 的 import 之前设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

# 注意：这里不再在顶部 import rag，否则会连带加载 torch/sentence-transformers，
# 导致网页一启动就黑屏干等。rag 推迟到 retrieve_docs 被调用（首次提问）时才加载。


# 长期记忆（用文件持久化，关掉重开也记得）
MEMORY_FILE = "capstone_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try: return __import__("json").load(open(MEMORY_FILE, encoding="utf-8"))
        except Exception: return []
    return []

def save_memory_entry(text):
    import json
    mem = load_memory()
    mem.append(text)
    json.dump(mem, open(MEMORY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

from langchain_core.tools import tool

@tool
def remember(text: str) -> str:
    """把关于用户的重要信息（姓名、偏好、事实）记录到长期记忆。"""
    save_memory_entry(text)
    return "已记住：" + text


@tool
def retrieve_docs(query: str) -> str:
    """当用户的问题需要依据"已提供的文档"来回答时，调用本工具检索相关段落。"""
    from rag import load_index   # 首次调用时才加载 rag（含 torch），避免启动黑屏
    vs = load_index()
    if vs is None:
        return "（知识库为空，请先往 docs/ 放入文档并运行建索引）"
    results = vs.similarity_search(query, k=3)
    out = []
    for i, r in enumerate(results, 1):
        src = r.metadata.get("source", "未知来源")
        out.append(f"[片段{i} 来源:{os.path.basename(src)}]\n{r.page_content}")
    return "\n\n".join(out)


TOOLS = [retrieve_docs, remember]


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_system_prompt() -> str:
    mem = load_memory()
    mem_text = "\n".join(f"  - {m}" for m in mem) if mem else "  （暂无）"
    return (
        "你是一个文档问答助手。回答必须基于检索到的文档内容，"
        "并在句末用[来源:文件名]标注出处；如果文档里没有相关信息，就老实说不知道。\n"
        "已知用户长期记忆：\n" + mem_text
    )


def build_graph(api_key: str | None = None):
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未提供 API Key：请在网页侧边栏输入，或设置环境变量 DASHSCOPE_API_KEY")
    model = ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )
    model_with_tools = model.bind_tools(TOOLS)

    def agent(state: State):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    tool_node = ToolNode(TOOLS)
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


def init_agent(api_key: str | None = None):
    """供网页界面调用：只编译图（模型/索引延迟到首次提问时由 retrieve_docs 懒加载），
    这样网页能秒开，不用在启动时干等模型载入。"""
    return build_graph(api_key=api_key)


def chat_once(graph, history, user_input: str) -> str:
    """供网页界面调用：发一条消息，返回 AI 回复字符串。"""
    messages = [SystemMessage(content=build_system_prompt())]
    messages.extend(history)
    messages.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": messages})
    return result["messages"][-1].content


def main():
    # 命令行模式：交互式读取 Key（网页模式不会走到这里，避免卡死）
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("请输入阿里云 DashScope API Key: ").strip()
    graph = init_agent(api_key=api_key)
    print("文档问答 Agent 已就绪。直接提问，它会先查 docs/ 再回答。")
    print("（输入 quit 退出）\n")
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        result = graph.invoke({
            "messages": [
                SystemMessage(content=build_system_prompt()),
                HumanMessage(content=user_input),
            ]
        })
        print("\nAI:", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()
