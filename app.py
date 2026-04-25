import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from pathlib import Path

st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="wide")

DATA_PATH = Path("data/sentiment_data.csv")


@st.cache_resource(show_spinner="Lade Sentiment-Modell...")
def load_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=["text", "label"])


def main():
    st.title("🧠 Sentiment Analysis Web App")
    st.caption("Analysiere Texte mit HuggingFace Transformers — gebaut mit Streamlit & Docker.")

    classifier = load_pipeline()

    tab1, tab2 = st.tabs(["✍️ Live-Analyse", "📊 Dataset"])

    with tab1:
        text = st.text_area(
            "Gib einen Text ein:",
            placeholder="z.B. I absolutely love this product!",
            height=140,
        )
        if st.button("Analysieren", type="primary"):
            if not text.strip():
                st.warning("Bitte einen Text eingeben.")
            else:
                result = classifier(text)[0]
                label = result["label"]
                score = result["score"]
                emoji = "😊" if label == "POSITIVE" else "😞"
                col1, col2 = st.columns(2)
                col1.metric("Sentiment", f"{emoji} {label}")
                col2.metric("Confidence", f"{score:.2%}")
                st.progress(float(score))

    with tab2:
        df = load_data()
        if df.empty:
            st.info("Keine Dummy-Daten gefunden. Notebook ausführen oder CSV hinzufügen.")
            return

        st.subheader("Dataset Vorschau")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Sentiment-Verteilung")
        fig, ax = plt.subplots(figsize=(6, 4))
        df["label"].value_counts().plot(
            kind="bar", ax=ax, color=["#4CAF50", "#F44336", "#FFC107"]
        )
        ax.set_xlabel("Label")
        ax.set_ylabel("Anzahl")
        ax.set_title("Sentiment Distribution")
        plt.xticks(rotation=0)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
