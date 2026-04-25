"""
Sentiment Analysis Web App
--------------------------
Streamlit-Frontend für ein HuggingFace Sentiment-Modell (DistilBERT SST-2).

Run:
    streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline

# ─────────────────────────────────────────────────────────────
# Konstanten
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
EXAMPLE_TEXT = "I love this product! It works flawlessly and made my day."

LABEL_MAP = {
    "POSITIVE": "Positive",
    "NEGATIVE": "Negative",
}

LABEL_COLORS = {
    "Positive": "#4CAF50",
    "Negative": "#F44336",
}


# ─────────────────────────────────────────────────────────────
# Modell laden (gecached über die ganze Session)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Lade Sentiment-Modell ...")
def load_model():
    """Lädt die HuggingFace Sentiment-Pipeline einmalig."""
    return pipeline("sentiment-analysis", model=MODEL_NAME)


# ─────────────────────────────────────────────────────────────
# Analyse-Logik
# ─────────────────────────────────────────────────────────────
def analyze_text(classifier, text: str) -> dict:
    """
    Führt die Sentiment-Analyse aus und liefert ein Dict mit
    label, confidence sowie Scores für beide Klassen.
    """
    result = classifier(text)[0]
    raw_label = result["label"]
    confidence = float(result["score"])
    label = LABEL_MAP.get(raw_label, raw_label)

    # Score auf beide Klassen aufteilen (Pipeline liefert nur den Top-Score)
    if label == "Positive":
        scores = {"Positive": confidence, "Negative": 1 - confidence}
    else:
        scores = {"Negative": confidence, "Positive": 1 - confidence}

    return {"label": label, "confidence": confidence, "scores": scores}


# ─────────────────────────────────────────────────────────────
# UI-Komponenten
# ─────────────────────────────────────────────────────────────
def render_result(result: dict) -> None:
    """Zeigt Label, Confidence-Metrik und Bar-Chart."""
    label = result["label"]
    confidence = result["confidence"]

    col1, col2 = st.columns(2)
    col1.metric("Sentiment", label)
    col2.metric("Confidence", f"{confidence:.2%}")

    st.progress(confidence)
    render_chart(result["scores"])


def render_chart(scores: dict) -> None:
    """Bar-Chart mit Positive- und Negative-Scores."""
    labels = list(scores.keys())
    values = list(scores.values())
    colors = [LABEL_COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Sentiment Scores")

    # Score-Werte über jeden Balken schreiben
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Sentiment Analysis Web App",
        layout="centered",
    )

    # ── Titel ────────────────────────────────────────────────
    st.title("Sentiment Analysis Web App")
    st.caption(
        "Analysiere englische Texte mit einem vortrainierten HuggingFace Modell "
        "(DistilBERT, fine-tuned auf SST-2)."
    )
    st.divider()

    # Modell laden — bei Fehler frühzeitig abbrechen
    try:
        classifier = load_model()
    except Exception as exc:  # noqa: BLE001
        st.error("Model could not be loaded")
        st.exception(exc)
        st.stop()

    # ── Input Bereich ────────────────────────────────────────
    st.subheader("Input")

    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    # Beispiel-Button MUSS vor dem Textarea kommen,
    # damit ein Klick den State aktualisiert bevor das Widget rendert.
    col_example, _ = st.columns([1, 3])
    if col_example.button("Load Example Text", use_container_width=True):
        st.session_state.user_text = EXAMPLE_TEXT

    text = st.text_area(
        "Dein Text:",
        key="user_text",
        height=160,
        placeholder="Tippe hier einen englischen Satz ein ...",
    )

    analyze_clicked = st.button(
        "Analyze Sentiment", type="primary", use_container_width=True
    )

    st.divider()

    # ── Ergebnis Bereich ─────────────────────────────────────
    st.subheader("Ergebnis")

    if not analyze_clicked:
        st.info("Gib einen Text ein und klicke auf 'Analyze Sentiment'.")
        return

    if not text.strip():
        st.warning("Bitte zuerst einen Text eingeben.")
        return

    with st.spinner("Analysiere ..."):
        try:
            result = analyze_text(classifier, text)
        except Exception as exc:  # noqa: BLE001
            st.error("Fehler bei der Analyse.")
            st.exception(exc)
            return

    render_result(result)


if __name__ == "__main__":
    main()
