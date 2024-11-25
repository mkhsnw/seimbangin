python -m venv seimbangin
seimbangin/Scripts/activate
pip install -r requirements.txt
uvicorn seimbangin_api:app --reload