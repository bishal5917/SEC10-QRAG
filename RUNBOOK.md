# RunBook

## macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
cp .env.example .env          # then paste your Gemini API key into .env
python3 ingest.py
python3 query.py
```

## Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env         # then paste your Gemini API key into .env
python ingest.py
python query.py
```
