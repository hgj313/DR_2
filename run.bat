@echo off
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
set PYTHONPATH=src
.venv\Scripts\python.exe src\interfaces\cli\main.py