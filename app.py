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

# Einfache Wortlisten für die Erklärung (keine ML, nur Heuristik)
POSITIVE_WORDS = {
    "love", "great", "amazing", "awesome", "excellent", "fantastic",
    "wonderful", "best", "good", "happy", "perfect", "brilliant",
    "enjoy", "enjoyed", "nice", "beautiful", "favorite", "recommend",
    "outstanding", "superb", "delightful", "pleased", "impressive",
}
NEGATIVE_WORDS = {
    "hate", "bad", "terrible", "awful", "worst", "horrible",
    "poor", "disappointing", "disappointed", "broken", "useless",
    "boring", "annoying", "sad", "angry", "waste", "scam", "garbage",
    "defective", "cheap", "ugly", "slow", "problem", "issue",
}

# Wort-Mappings für die Transformation (sehr einfache Heuristik)
TO_POSITIVE = {
    "bad": "great", "terrible": "wonderful", "awful": "amazing",
    "worst": "best", "horrible": "delightful", "poor": "excellent",
    "disappointing": "impressive", "disappointed": "pleased",
    "broken": "flawless", "useless": "useful", "boring": "exciting",
    "annoying": "pleasant", "sad": "happy", "angry": "calm",
    "waste": "treasure", "garbage": "gem", "defective": "perfect",
    "cheap": "high-quality", "ugly": "beautiful", "slow": "fast",
    "problem": "feature", "issue": "highlight", "hate": "love",
    "not": "really", "never": "always", "no": "yes",
}
TO_NEGATIVE = {
    "love": "hate", "great": "terrible", "amazing": "awful",
    "awesome": "horrible", "excellent": "poor", "fantastic": "disappointing",
    "wonderful": "boring", "best": "worst", "good": "bad",
    "happy": "sad", "perfect": "broken", "brilliant": "dull",
    "enjoy": "endure", "enjoyed": "endured", "nice": "annoying",
    "beautiful": "ugly", "favorite": "least favorite",
    "recommend": "warn against", "outstanding": "mediocre",
    "superb": "awful", "delightful": "tedious", "pleased": "frustrated",
    "impressive": "underwhelming", "always": "never", "really": "barely",
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
# Erklärungs-Logik (einfache Keyword-Heuristik)
# ─────────────────────────────────────────────────────────────
def find_keywords(text: str) -> tuple[list[str], list[str]]:
    """Sucht positive und negative Schlüsselwörter im Text."""
    tokens = {t.strip(".,!?;:\"'()[]").lower() for t in text.split()}
    positives = sorted(tokens & POSITIVE_WORDS)
    negatives = sorted(tokens & NEGATIVE_WORDS)
    return positives, negatives


def build_explanation(label: str, confidence: float,
                      positives: list[str], negatives: list[str]) -> str:
    """Baut eine kurze, menschenlesbare Erklärung."""
    strength = (
        "sehr sicher" if confidence >= 0.95
        else "ziemlich sicher" if confidence >= 0.80
        else "eher unsicher"
    )
    matching = positives if label == "Positive" else negatives
    counter = negatives if label == "Positive" else positives

    if matching:
        words = ", ".join(f"\"{w}\"" for w in matching)
        base = (
            f"Der Text wurde als **{label}** eingestuft, weil er Wörter mit "
            f"{'positiver' if label == 'Positive' else 'negativer'} "
            f"Tonalität enthält ({words})."
        )
    else:
        base = (
            f"Der Text wurde als **{label}** eingestuft. Die Bewertung "
            f"ergibt sich aus dem Gesamtkontext, nicht aus einzelnen Schlüsselwörtern."
        )

    if counter:
        base += (
            f" Auch wenn Begriffe wie {', '.join(repr(w) for w in counter)} "
            f"in die andere Richtung deuten, überwiegt die "
            f"{label.lower()} Tonalität."
        )

    base += f" Das Modell ist sich {strength} (Confidence {confidence:.0%})."
    return base


def transform_text(text: str, target: str) -> str:
    """
    Formt den Text in Richtung des Ziel-Sentiments um.
    Einfache Heuristik: Wort-Substitution + Prefix/Suffix.
    """
    mapping = TO_POSITIVE if target == "Positive" else TO_NEGATIVE
    out = []
    for token in text.split():
        prefix = ""
        suffix = ""
        # Satzzeichen vom Wort trennen
        while token and token[0] in "\"'([":
            prefix += token[0]
            token = token[1:]
        while token and token[-1] in ".,!?;:\"')]":
            suffix = token[-1] + suffix
            token = token[:-1]
        clean = token.lower()
        if clean in mapping:
            replacement = mapping[clean]
            # Großschreibung beibehalten
            if token[:1].isupper():
                replacement = replacement.capitalize()
            out.append(prefix + replacement + suffix)
        else:
            out.append(prefix + token + suffix)

    rewritten = " ".join(out)

    if target == "Positive":
        return f"{rewritten} Overall, it's a really pleasant experience!"
    else:
        return f"{rewritten} Overall, it's quite a disappointing experience."


def highlight_text(text: str, positives: list[str], negatives: list[str]) -> str:
    """Markiert Schlüsselwörter farbig in einer Markdown-Version des Textes."""
    out = []
    for token in text.split():
        clean = token.strip(".,!?;:\"'()[]").lower()
        if clean in positives:
            out.append(f":green[**{token}**]")
        elif clean in negatives:
            out.append(f":red[**{token}**]")
        else:
            out.append(token)
    return " ".join(out)


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

    target_sentiment = st.selectbox(
        "Target Sentiment",
        ["Positive", "Negative"],
        key="target_sentiment",
        help="In welche Richtung soll der Text später verändert werden?",
    )
    st.caption(f"Ausgewähltes Ziel-Sentiment: **{target_sentiment}**")

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

    # ── Explanation Bereich ──────────────────────────────────
    st.divider()
    st.subheader("Explanation")

    positives, negatives = find_keywords(text)
    explanation = build_explanation(
        result["label"], result["confidence"], positives, negatives
    )
    st.markdown(explanation)

    if positives or negatives:
        st.markdown("**Hervorgehobener Text:**")
        st.markdown(highlight_text(text, positives, negatives))

        col_pos, col_neg = st.columns(2)
        col_pos.markdown(
            "**Positive Wörter:** "
            + (", ".join(f":green[{w}]" for w in positives) if positives else "_keine_")
        )
        col_neg.markdown(
            "**Negative Wörter:** "
            + (", ".join(f":red[{w}]" for w in negatives) if negatives else "_keine_")
        )
    else:
        st.caption(
            "Hinweis: Es wurden keine bekannten Schlüsselwörter erkannt. "
            "Die Bewertung basiert auf dem Gesamtkontext des Modells."
        )

    # ── Transformation Bereich ───────────────────────────────
    st.divider()
    st.subheader("Transformierter Text")

    target = st.session_state.get("target_sentiment", "Positive")
    transformed = transform_text(text, target)

    tone = "freundlich, optimistisch" if target == "Positive" else "kritisch, pessimistisch"
    st.caption(f"Ziel-Sentiment: **{target}** ({tone})")
    st.markdown(f"> {transformed}")

    st.caption(
        "Hinweis: Diese Umformulierung basiert auf einer einfachen "
        "Wort-für-Wort-Heuristik, nicht auf einem Sprachmodell."
    )


if __name__ == "__main__":
    main()
