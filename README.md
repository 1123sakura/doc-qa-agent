# 文档问答 Agent（agent2）

一个能「读懂你给的文档、先检索再回答、带引用来源」的 AI Agent。
技术栈：**DeepSeek（大模型）+ LangGraph（编排）+ FAISS（向量库）+ bge 中文嵌入 + Streamlit（网页）**。

## 一、项目结构

| 文件 | 作用 | 你要动吗 |
|---|---|---|
| **app.py** | 网页入口（Streamlit） | 运行它 |
| agent.py | LangGraph Agent 编排（被 app.py 调用） | 不用改 |
| rag.py | 文档检索内核 RAG（被 agent.py 调用） | 不用改 |
| docs/ | 把你的 .txt/.md/.pdf 文档放这里 | 换内容时动 |
| requirements.txt | 依赖清单 | 装依赖时用 |
| .env | 存 DeepSeek Key（已被 gitignore，绝不外传） | 填你自己的 Key |
| start.bat | 双击即用项目 venv 启动 | 懒人用 |

## 二、本地运行（3 选 1）

**前提**：`.env` 里已写好 `DEEPSEEK_API_KEY=sk-你的key`（程序自动读取，不用每次手填）。

1. **PyCharm 绿色三角（推荐）**：Run → Edit Configurations → ＋Python → 模块名称填 `streamlit` → 参数填 `run app.py` → 工作目录选项目根目录 → 解释器选 `agent2/.venv` → 确定 → 点绿色三角 ▶
2. **双击 start.bat**：在文件夹里双击，自动用 `.venv` 启动
3. **终端**：在 `.venv` 激活时执行 `streamlit run app.py`

启动后按住 Ctrl 点 `http://localhost:8501` 打开网页即可提问（如「年假怎么请」）。

> 首次提问会加载中文嵌入模型（已缓存到 `~/.cache/huggingface`，仅第一次约 15–25 秒），之后流畅。打开网页本身是秒开的。

## 三、换成你自己的文档

把 `.txt` / `.md` / `.pdf` 丢进 `docs/`，删除 `faiss_index` 文件夹，重新运行即可重建索引。
**建议换成你目标岗位相关的真实材料**（面试/产品/制度文档等），问答才有简历价值。

## 四、工作原理（面试常问）

- **检索（RAG）**：用 TextLoader/PyPDFLoader 加载文档 → RecursiveCharacterTextSplitter 切片 → `BAAI/bge-small-zh-v1.5` 模型转成向量 → 存入 FAISS 向量库。
- **编排（LangGraph）**：用状态图（StateGraph）管理「思考→调工具→再思考」的循环；`retrieve_docs` 工具检索最相关的前 3 段，`remember` 工具把重要信息写入长期记忆。
- **生成**：DeepSeek 基于检索到的文档片段作答，避免凭空编造（幻觉）。

## 五、部署上线（拿公网 URL，写进简历用）

> 部署是把项目从「本地自玩」变成「可展示作品」的关键一步。

1. **先把 git 身份改成你自己的**（否则推到 GitHub 后 commit 不显示你的头像）：
   ```bash
   git config user.name  "你的GitHub用户名"
   git config user.email "你的GitHub邮箱"
   ```
2. **推到 GitHub**：
   - 在 GitHub 新建一个**公开**仓库（如 `doc-qa-agent`）
   - 本地执行：
     ```bash
     git remote add origin https://github.com/你的用户名/你的仓库.git
     git push -u origin main
     ```
3. **Streamlit Cloud 部署**：打开 https://share.streamlit.io → 用 GitHub 登录 → New app → 选刚才的仓库、主文件填 `app.py` → 展开 Advanced settings：
   - **Secrets** 填：`DEEPSEEK_API_KEY = "你的key"`
   - **Environment variables** 填：`HF_ENDPOINT = https://huggingface.co`（海外服务器用官方源，别用国内镜像）
   - 点 Deploy → 等 1–3 分钟 → 拿到公网 URL，写进简历。

**部署注意**：
- 仓库不含 `faiss_index`，部署后首次提问时会自动构建/加载索引（需联网下载嵌入模型，稍慢属正常）。
- `torch` / `sentence-transformers` 体积较大，Streamlit Cloud 安装依赖可能比本地慢，耐心等。
