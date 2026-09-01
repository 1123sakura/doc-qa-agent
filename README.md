# 文档问答 Agent（agent2）

一个能「读你给的文档、先检索再回答、带引用来源」的 AI Agent。
技术栈：DeepSeek + LangGraph + FAISS + bge 中文嵌入 + Streamlit。

## 你只需要关心一个文件

| 文件 | 作用 | 你要动吗 |
|---|---|---|
| **app.py** | **网页入口**，右键 Run 就能用 | 运行它 |
| agent.py | LangGraph Agent（被 app.py 调用） | 不用改 |
| rag.py | 文档检索内核（被 agent.py 调用） | 不用改 |
| docs/ | 把你的 .txt/.md/.pdf 文档放这里 | 换内容时动 |
| requirements.txt | 依赖清单 | 装依赖时用 |

## 在 PyCharm 里运行（3 步）

1. **打开项目**：PyCharm → File → Open → 选 `agent2` 这个文件夹
2. **配解释器**（关键，否则 import 报错）：
   - 推荐直接用你已装好依赖的 Python：
     File → Settings → Python Interpreter → 选 `D:\develop\python\python.exe`
   - 如果你让 PyCharm 新建了 venv，就在底部 **Terminal** 里跑：
     `pip install -r requirements.txt`
3. **运行**：左侧选中 `app.py` → 右键 → **Run 'app.py'**
   - 浏览器自动打开 `http://localhost:8501`
   - 左侧填你的 **DeepSeek API Key**
   - 聊天框输入问题，例如「年假怎么请」
   - 首次会下载中文嵌入模型（约 100MB，需 1~2 分钟），耐心等

## 换成你自己的文档

把 `.txt` / `.md` / `.pdf` 丢进 `docs/`，删除 `faiss_index` 文件夹（如果已生成），
重新 Run `app.py` 即可重建索引。

## 部署上线（拿公网网址，写进简历用）

git 推到 GitHub 公开仓库 → 打开 share.streamlit.io → 用 GitHub 登录 →
New app → 选仓库、主文件填 `app.py` → Advanced settings 里：
- Secrets 填 `DEEPSEEK_API_KEY = "你的key"`
- Environment variables 填 `HF_ENDPOINT = https://huggingface.co`
→ Deploy → 等 1~3 分钟 → 拿到公网 URL。
