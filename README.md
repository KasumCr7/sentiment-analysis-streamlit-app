# sentiment-analysis-streamlit-app

Streamlit-App für Sentiment-Analyse englischer Texte. Verwendet ein vortrainiertes
DistilBERT-Modell (SST-2) über die HuggingFace Transformers Pipeline. Containerisiert
mit Docker.

## Features

- Sentiment-Klassifikation (Positive / Negative) mit Confidence-Score
- Bar-Chart der beiden Klassen-Scores
- Beispieltext per Knopfdruck einfügbar
- Modell wird zur Laufzeit gecacht (`@st.cache_resource`)
- Eingabevalidierung und Fehlerbehandlung beim Modell-Load

## Tech Stack

- Python 3.11
- Streamlit
- HuggingFace Transformers (`distilbert-base-uncased-finetuned-sst-2-english`)
- PyTorch
- pandas, matplotlib
- Docker / Docker Compose

## Projektstruktur

```
sentiment-analysis-app/
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── data/
│   └── sentiment_data.csv
└── notebooks/
    └── analysis.ipynb
```

## Lokale Ausführung

Voraussetzungen: Python 3.11+, pip.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

App läuft auf http://localhost:8501.

Erster Start lädt das Modell (~250 MB) in den HuggingFace-Cache.

## Docker-Ausführung

Voraussetzungen: Docker 20.10+, Docker Compose v2.

```bash
docker compose up --build
```

App läuft auf http://localhost:8501.

Stoppen:

```bash
docker compose down
```

Das Verzeichnis `data/` ist als Volume gemountet und persistiert zwischen Container-Restarts.

## Konfiguration

| Variable             | Default | Zweck                              |
|----------------------|---------|------------------------------------|
| `PYTHONUNBUFFERED`   | `1`     | Logs sofort an stdout              |

## Lizenz

MIT
