# -*- coding: utf-8 -*-
"""
文档问答 Agent · RAG 内核

把「文档 → 切片 → 向量索引 → 检索工具」这一套做出来。
这是整个项目最核心的能力：让 Agent 能"读"你给的文档再回答。

运行：python rag.py   # 扫描 docs/ 建索引，然后让你提问测试检索
"""
import os
import glob

# HuggingFace 在国内常被墙，先切到国内镜像；必须在任何 huggingface_hub 相关 import 之前设置。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 固定缓存目录：保证 uv run 和 PyCharm 运行配置命中同一份模型缓存，
# 避免每次启动都重新下载 bge 模型（~90MB），这是加载慢的主因。
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

DOCS_DIR = "docs"            # 把你的文档（.txt/.md/.pdf）放进这个文件夹
INDEX_DIR = "faiss_index"    # 建好的向量索引会存这里，下次直接加载
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"   # 中文友好、本地运行、体积小

# 缓存 embeddings 实例，避免每次提问都重新加载模型（否则网页版会很慢）
_EMBEDDINGS_CACHE = None
# 缓存向量库实例，避免每次提问都从磁盘重载 FAISS 索引（否则每次问答都要读盘）
_VECTORSTORE_CACHE = None


def load_documents(folder: str = DOCS_DIR):
    """读取 docs/ 下所有 .txt/.md/.pdf，返回一个 Document 列表。"""
    docs = []
    paths = glob.glob(os.path.join(folder, "**", "*"), recursive=True)
    for p in paths:
        if p.lower().endswith((".txt", ".md")):
            docs.extend(TextLoader(p, encoding="utf-8").load())
        elif p.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(p).load())
    print(f"已加载 {len(docs)} 个文档片段（按文件计）")
    return docs


def split_documents(docs, chunk_size: int = 400, chunk_overlap: int = 50):
    """把长文档切成小块。RAG 的关键：块太大检索不准，太小丢上下文。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def get_embeddings():
    # normalize_embeddings=True 是 bge 模型的要求，能提升检索质量
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        print("[首次加载嵌入模型，请稍候...]")
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS_CACHE


def build_index():
    """扫描 docs/，切片 + 嵌入 + 建 FAISS 索引，保存到本地。"""
    docs = load_documents()
    if not docs:
        print("docs/ 里没有文档，先放几个 .txt/.md/.pdf 再运行。")
        return None
    chunks = split_documents(docs)
    print(f"切成 {len(chunks)} 个 chunk")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    global _VECTORSTORE_CACHE
    _VECTORSTORE_CACHE = vectorstore
    print(f"索引已保存到 {INDEX_DIR}")
    return vectorstore


def load_index():
    global _VECTORSTORE_CACHE
    if _VECTORSTORE_CACHE is not None:
        return _VECTORSTORE_CACHE
    if not os.path.exists(INDEX_DIR):
        return build_index()
    embeddings = get_embeddings()
    _VECTORSTORE_CACHE = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return _VECTORSTORE_CACHE


@tool
def retrieve_docs(query: str) -> str:
    """当用户的问题需要依据"已提供的文档"来回答时，调用本工具检索相关段落。"""
    vs = load_index()
    if vs is None:
        return "（知识库为空，请先往 docs/ 放入文档并运行建索引）"
    results = vs.similarity_search(query, k=3)
    # 把命中的 chunk 拼成文本返回给模型，并标明来源文件名
    out = []
    for i, r in enumerate(results, 1):
        src = r.metadata.get("source", "未知来源")
        out.append(f"[片段{i} 来源:{os.path.basename(src)}]\n{r.page_content}")
    return "\n\n".join(out)


if __name__ == "__main__":
    # 直接运行 = 建索引 + 手动测试检索效果
    vs = load_index()
    if vs is None:
        exit()
    while True:
        q = input("测试检索（输入 quit 退出）: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        hits = vs.similarity_search(q, k=3)
        for i, h in enumerate(hits, 1):
            print(f"\n[{i}] 来源:{os.path.basename(h.metadata.get('source','?'))}")
            print(h.page_content[:200])
