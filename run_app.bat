@echo off
python -m pip install -r requirements.txt
python -m pip install -e .
streamlit run app\app.py
