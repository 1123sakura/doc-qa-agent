@echo off
REM 文档问答 Agent · 一键启动脚本
REM 用法：把这个文件放在项目根目录，双击即可启动 Streamlit 网页
cd /d %~dp0
echo ========================================
echo   文档问答 Agent 启动中...
echo   启动后请按住 Ctrl 点击终端里的 http://localhost:8501
echo   关闭程序：在窗口里按 Ctrl+C，或直接关掉窗口
echo ========================================
echo.
.venv\Scripts\python.exe -m streamlit run app.py
pause
