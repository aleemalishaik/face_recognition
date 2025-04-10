@echo off
python -m venv venv
call venv\Scripts\activate
pip install --no-index --find-links=packages -r requirements.txt
