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
## Langfuse observability (optional — same Docker command on macOS & Windows)

```bash
docker compose -f docker-compose.langfuse.yml up -d   # start Langfuse v3 (needs Rancher/Docker Desktop)
# open http://localhost:3000 → sign up → create project → Settings → API Keys → create
# paste LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY into .env
# set enable_langfuse = True in app/core/config.py
# then run a query — traces appear in the Langfuse UI:
#   macOS:    python3 query.py -q "What was Apple's revenue in Q3 2023?"
#   Windows:  python  query.py -q "What was Apple's revenue in Q3 2023?"
docker compose -f docker-compose.langfuse.yml down    # stop when done
```
