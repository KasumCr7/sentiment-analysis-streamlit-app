<<<<<<< HEAD
# 🧠 Sentiment Analysis Web App

Eine interaktive Web-App zur Sentiment-Analyse von Texten — gebaut mit **Streamlit**, **HuggingFace Transformers** und vollständig in **Docker** containerisiert. Kommt mit einem Jupyter-Notebook für die explorative Datenanalyse.

---

## ✨ Features

- **Live Sentiment-Analyse** beliebiger Texte über DistilBERT (SST-2)
- **Dataset-Explorer** mit pandas-Vorschau und Verteilungs-Plot
- **Jupyter Notebook** für EDA, Visualisierung & Modell-Evaluation
- **Dockerized** — App + Jupyter laufen über `docker compose up`
- **HuggingFace Cache** persistiert per Volume (kein Re-Download bei Restart)
- Funktioniert offline mit mitgeliefertem Dummy-Datensatz, optional Kaggle-Daten einlegbar

---

## 🗂️ Projektstruktur

```
sentiment-analysis-app/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app.py                    # Streamlit App
├── notebooks/
│   └── analysis.ipynb        # Datenanalyse & HF-Pipeline-Test
├── data/
│   └── sentiment_data.csv    # Dummy-Daten (30 Beispiele)
└── README.md
```

---

## 📋 Anforderungen

- **Docker** ≥ 20.10
- **Docker Compose** ≥ 2.0
- ~3 GB freier Speicher (für Modell-Download beim ersten Start)

Ohne Docker:
- Python 3.11+
- pip

---

## 🚀 Installation

### Option A — Docker (empfohlen)

```bash
git clone <repo-url>
cd sentiment-analysis-app
docker compose up --build
```

Beim ersten Start lädt HuggingFace das DistilBERT-Modell (~250 MB). Danach im Cache.

### Option B — Lokal mit Python

```bash
cd sentiment-analysis-app
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🖥️ Usage

Nach `docker compose up`:

| Service | URL | Zweck |
|---|---|---|
| Streamlit App | http://localhost:8501 | Live-Analyse + Dataset-Viewer |
| Jupyter Notebook | http://localhost:8888 | `notebooks/analysis.ipynb` öffnen |

### App-Tabs

1. **✍️ Live-Analyse** — Text eingeben → Sentiment + Confidence + Progress-Bar
2. **📊 Dataset** — Vorschau der CSV + Bar-Chart der Label-Verteilung

### Eigene Daten verwenden

Lege eine CSV mit den Spalten `text,label` unter `data/sentiment_data.csv` ab. Optional kannst du `data/kaggle_sentiment.csv` hinterlegen — das Notebook nutzt diese automatisch.

---

## 🛑 Stoppen

```bash
docker compose down
```

Cache & Volumes löschen:

```bash
docker compose down -v
```

---

## 🧪 Tech Stack

| Komponente | Verwendung |
|---|---|
| Streamlit | Web-UI |
| HuggingFace Transformers | DistilBERT-Pipeline |
| PyTorch | Backend für das Modell |
| pandas | Datenverarbeitung |
| matplotlib | Visualisierungen |
| Jupyter | EDA-Notebook |
| Docker | Reproduzierbares Deployment |

---

## 📝 Lizenz

MIT — frei zu nutzen, anzupassen und weiterzuentwickeln.
=======
# sentiment-analysis-streamlit-app
Simple Sentiment Analysis web App using Streamlit, Transformers, and Docker.
>>>>>>> a823594900a67302045ae1f26cfd7f90f55a0f9b
