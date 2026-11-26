@echo off
rem One-step Hugging Face publish: asks for your token, then uploads the model and the demo Space.
cd /d "%~dp0.."
set "PYTHONPATH=D:\Projects\image-proccessing-project\.venv\Lib\site-packages"
set "PYTHONIOENCODING=utf-8"
"C:\Users\User\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe" deploy\publish.py %*
pause
