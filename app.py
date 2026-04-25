"""
Sentiment Analysis Web App
--------------------------
Streamlit-Frontend für ein multilinguales HuggingFace Sentiment-Modell
(nlptown/bert-base-multilingual-uncased-sentiment).

Run:
    streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline

# ─────────────────────────────────────────────────────────────
# Konstanten
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

LANGUAGE_OPTIONS: tuple[str, ...] = ("English", "Deutsch", "Français")
TARGET_SENTIMENT_OPTIONS: tuple[str, ...] = ("Positive", "Negative")

LABEL_COLORS = {
    "Positive": "#4CAF50",
    "Negative": "#F44336",
    "Neutral": "#9E9E9E",
}

# ─────────────────────────────────────────────────────────────
# Zentrale Sprachsteuerung — alle UI-Strings pro Sprache
# ─────────────────────────────────────────────────────────────
TEXTS = {
    "English": {
        "title": "Sentiment Analysis Web App",
        "subtitle": (
            "Analyze texts in multiple languages with a multilingual "
            "HuggingFace BERT model (nlptown, 1–5 stars)."
        ),
        "language_label": "Language",
        "input_section": "Input",
        "input_label": "Your text:",
        "input_placeholder": "Type a sentence here ...",
        "load_example": "Load Example Text",
        "target_label": "Target Sentiment",
        "target_help": "In which direction should the text be rewritten later?",
        "target_invalid": "Invalid target sentiment. Please choose an option from the list.",
        "target_selected": "Selected target sentiment:",
        "analyze_button": "Analyze Sentiment",
        "no_text_hint": "Please enter a text first to enable the analysis.",
        "result_section": "Result",
        "result_placeholder": "Enter a text and click 'Analyze Sentiment'.",
        "no_text_warning": "Please enter a text first.",
        "spinner": "Analyzing text — this may take a few seconds ...",
        "analysis_error": (
            "An error occurred during analysis. Please try again with a different text."
        ),
        "model_error": (
            "The sentiment model could not be loaded. "
            "Please check your internet connection and restart the app."
        ),
        "tech_details": "Technical details",
        "success": "Analysis complete ({language}) — sentiment detected as **{label}** (confidence {conf}).",
        "metric_sentiment": "Sentiment",
        "metric_confidence": "Confidence",
        "metric_stars": "Stars",
        "chart_xlabel": "Class",
        "chart_ylabel": "Probability",
        "explanation_section": "Explanation",
        "explanation_with_words": (
            "The text was classified as **{label}** because it contains words "
            "with {tone} tonality ({words})."
        ),
        "explanation_no_words": (
            "The text was classified as **{label}**. The rating comes from the "
            "overall context, not from individual keywords."
        ),
        "explanation_counter": (
            " Even though terms like {words} point in the other direction, the "
            "{label_lower} tonality dominates."
        ),
        "explanation_confidence": " The model is {strength} (confidence {conf}).",
        "tone_positive": "positive",
        "tone_negative": "negative",
        "strength_high": "very confident",
        "strength_mid": "fairly confident",
        "strength_low": "rather uncertain",
        "highlighted": "**Highlighted text:**",
        "positive_words_label": "**Positive words:**",
        "negative_words_label": "**Negative words:**",
        "none": "_none_",
        "no_keywords_hint": (
            "Note: no known keywords were found. The rating is based on the "
            "model's overall context."
        ),
        "transform_section": "Transformed Text",
        "transform_caption": "Target sentiment: **{target}** ({tone})",
        "transform_tone_positive": "friendly, optimistic",
        "transform_tone_negative": "critical, pessimistic",
        "transform_hint": (
            "Note: this rewrite uses a simple word-by-word heuristic, "
            "not a language model."
        ),
        "transform_suffix_positive": " Overall, it's a really pleasant experience!",
        "transform_suffix_negative": " Overall, it's quite a disappointing experience.",
        "label_positive": "Positive",
        "label_negative": "Negative",
        "label_neutral": "Neutral",
        "example": "I love this product! It works flawlessly and made my day.",
    },
    "Deutsch": {
        "title": "Sentiment Analyse Web-App",
        "subtitle": (
            "Analysiere Texte in mehreren Sprachen mit einem multilingualen "
            "HuggingFace BERT-Modell (nlptown, 1–5 Sterne)."
        ),
        "language_label": "Sprache",
        "input_section": "Eingabe",
        "input_label": "Dein Text:",
        "input_placeholder": "Tippe hier einen Satz ein ...",
        "load_example": "Beispieltext laden",
        "target_label": "Ziel-Sentiment",
        "target_help": "In welche Richtung soll der Text später umgeschrieben werden?",
        "target_invalid": "Ungültiges Ziel-Sentiment. Bitte eine Option aus der Liste wählen.",
        "target_selected": "Ausgewähltes Ziel-Sentiment:",
        "analyze_button": "Sentiment analysieren",
        "no_text_hint": "Bitte zuerst einen Text eingeben, um die Analyse zu aktivieren.",
        "result_section": "Ergebnis",
        "result_placeholder": "Gib einen Text ein und klicke auf 'Sentiment analysieren'.",
        "no_text_warning": "Bitte zuerst einen Text eingeben.",
        "spinner": "Analysiere Text — das kann einige Sekunden dauern ...",
        "analysis_error": (
            "Bei der Analyse ist ein Fehler aufgetreten. "
            "Bitte mit einem anderen Text erneut versuchen."
        ),
        "model_error": (
            "Das Sentiment-Modell konnte nicht geladen werden. "
            "Bitte Internet-Verbindung prüfen und die App neu starten."
        ),
        "tech_details": "Technische Details",
        "success": "Analyse abgeschlossen ({language}) — Sentiment erkannt als **{label}** (Confidence {conf}).",
        "metric_sentiment": "Sentiment",
        "metric_confidence": "Konfidenz",
        "metric_stars": "Sterne",
        "chart_xlabel": "Klasse",
        "chart_ylabel": "Wahrscheinlichkeit",
        "explanation_section": "Erklärung",
        "explanation_with_words": (
            "Der Text wurde als **{label}** eingestuft, weil er Wörter mit "
            "{tone} Tonalität enthält ({words})."
        ),
        "explanation_no_words": (
            "Der Text wurde als **{label}** eingestuft. Die Bewertung ergibt "
            "sich aus dem Gesamtkontext, nicht aus einzelnen Schlüsselwörtern."
        ),
        "explanation_counter": (
            " Auch wenn Begriffe wie {words} in die andere Richtung deuten, "
            "überwiegt die {label_lower} Tonalität."
        ),
        "explanation_confidence": " Das Modell ist sich {strength} (Confidence {conf}).",
        "tone_positive": "positiver",
        "tone_negative": "negativer",
        "strength_high": "sehr sicher",
        "strength_mid": "ziemlich sicher",
        "strength_low": "eher unsicher",
        "highlighted": "**Hervorgehobener Text:**",
        "positive_words_label": "**Positive Wörter:**",
        "negative_words_label": "**Negative Wörter:**",
        "none": "_keine_",
        "no_keywords_hint": (
            "Hinweis: Es wurden keine bekannten Schlüsselwörter erkannt. "
            "Die Bewertung basiert auf dem Gesamtkontext des Modells."
        ),
        "transform_section": "Umgeschriebener Text",
        "transform_caption": "Ziel-Sentiment: **{target}** ({tone})",
        "transform_tone_positive": "freundlich, optimistisch",
        "transform_tone_negative": "kritisch, pessimistisch",
        "transform_hint": (
            "Hinweis: Diese Umformulierung basiert auf einer einfachen "
            "Wort-für-Wort-Heuristik, nicht auf einem Sprachmodell."
        ),
        "transform_suffix_positive": " Insgesamt ist es eine wirklich angenehme Erfahrung!",
        "transform_suffix_negative": " Insgesamt ist es eine ziemlich enttäuschende Erfahrung.",
        "label_positive": "Positiv",
        "label_negative": "Negativ",
        "label_neutral": "Neutral",
        "example": "Ich liebe dieses Produkt! Es funktioniert einwandfrei.",
    },
    "Français": {
        "title": "Application d'analyse de sentiment",
        "subtitle": (
            "Analysez des textes dans plusieurs langues avec un modèle BERT "
            "multilingue HuggingFace (nlptown, 1–5 étoiles)."
        ),
        "language_label": "Langue",
        "input_section": "Saisie",
        "input_label": "Votre texte :",
        "input_placeholder": "Saisissez une phrase ici ...",
        "load_example": "Charger un exemple",
        "target_label": "Sentiment cible",
        "target_help": "Dans quelle direction le texte doit-il être réécrit ?",
        "target_invalid": "Sentiment cible invalide. Veuillez choisir une option dans la liste.",
        "target_selected": "Sentiment cible sélectionné :",
        "analyze_button": "Analyser le sentiment",
        "no_text_hint": "Veuillez d'abord saisir un texte pour activer l'analyse.",
        "result_section": "Résultat",
        "result_placeholder": "Saisissez un texte et cliquez sur 'Analyser le sentiment'.",
        "no_text_warning": "Veuillez d'abord saisir un texte.",
        "spinner": "Analyse en cours — cela peut prendre quelques secondes ...",
        "analysis_error": (
            "Une erreur s'est produite lors de l'analyse. "
            "Veuillez réessayer avec un autre texte."
        ),
        "model_error": (
            "Le modèle de sentiment n'a pas pu être chargé. "
            "Veuillez vérifier votre connexion Internet et redémarrer l'application."
        ),
        "tech_details": "Détails techniques",
        "success": "Analyse terminée ({language}) — sentiment détecté comme **{label}** (confiance {conf}).",
        "metric_sentiment": "Sentiment",
        "metric_confidence": "Confiance",
        "metric_stars": "Étoiles",
        "chart_xlabel": "Classe",
        "chart_ylabel": "Probabilité",
        "explanation_section": "Explication",
        "explanation_with_words": (
            "Le texte a été classé comme **{label}** car il contient des mots "
            "à tonalité {tone} ({words})."
        ),
        "explanation_no_words": (
            "Le texte a été classé comme **{label}**. L'évaluation provient du "
            "contexte global et non de mots-clés spécifiques."
        ),
        "explanation_counter": (
            " Même si des termes comme {words} pointent dans l'autre direction, "
            "la tonalité {label_lower} l'emporte."
        ),
        "explanation_confidence": " Le modèle est {strength} (confiance {conf}).",
        "tone_positive": "positive",
        "tone_negative": "négative",
        "strength_high": "très confiant",
        "strength_mid": "assez confiant",
        "strength_low": "plutôt incertain",
        "highlighted": "**Texte mis en évidence :**",
        "positive_words_label": "**Mots positifs :**",
        "negative_words_label": "**Mots négatifs :**",
        "none": "_aucun_",
        "no_keywords_hint": (
            "Remarque : aucun mot-clé connu n'a été détecté. "
            "L'évaluation repose sur le contexte global du modèle."
        ),
        "transform_section": "Texte transformé",
        "transform_caption": "Sentiment cible : **{target}** ({tone})",
        "transform_tone_positive": "amical, optimiste",
        "transform_tone_negative": "critique, pessimiste",
        "transform_hint": (
            "Remarque : cette reformulation utilise une simple heuristique "
            "mot à mot, pas un modèle de langage."
        ),
        "transform_suffix_positive": " Dans l'ensemble, c'est une expérience vraiment agréable !",
        "transform_suffix_negative": " Dans l'ensemble, c'est une expérience plutôt décevante.",
        "label_positive": "Positif",
        "label_negative": "Négatif",
        "label_neutral": "Neutre",
        "example": "J'adore ce produit ! Il fonctionne parfaitement.",
    },
}

# ─────────────────────────────────────────────────────────────
# Wortlisten (Englisch — für Heuristik in Erklärung & Transformation)
# ─────────────────────────────────────────────────────────────
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
# Modell laden (gecached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading sentiment model ...")
def load_model():
    return pipeline("sentiment-analysis", model=MODEL_NAME)


# ─────────────────────────────────────────────────────────────
# Analyse-Logik
# ─────────────────────────────────────────────────────────────
def analyze_text(classifier, text: str, texts: dict) -> dict:
    """1–5 Sterne → Positive / Neutral / Negative."""
    result = classifier(text)[0]
    confidence = float(result["score"])
    stars = int(result["label"].split()[0])

    if stars >= 4:
        label = "Positive"
        scores = {"Positive": confidence, "Negative": 1 - confidence}
    elif stars <= 2:
        label = "Negative"
        scores = {"Negative": confidence, "Positive": 1 - confidence}
    else:
        label = "Neutral"
        scores = {"Positive": 0.5, "Negative": 0.5}

    display_label = texts[f"label_{label.lower()}"]
    return {
        "label": label,
        "display_label": display_label,
        "confidence": confidence,
        "scores": scores,
        "stars": stars,
    }


# ─────────────────────────────────────────────────────────────
# Erklärungs- und Transformations-Logik
# ─────────────────────────────────────────────────────────────
def find_keywords(text: str) -> tuple[list[str], list[str]]:
    tokens = {t.strip(".,!?;:\"'()[]").lower() for t in text.split()}
    return sorted(tokens & POSITIVE_WORDS), sorted(tokens & NEGATIVE_WORDS)


def build_explanation(label: str, confidence: float,
                      positives: list[str], negatives: list[str],
                      texts: dict) -> str:
    if confidence >= 0.95:
        strength = texts["strength_high"]
    elif confidence >= 0.80:
        strength = texts["strength_mid"]
    else:
        strength = texts["strength_low"]

    display_label = texts[f"label_{label.lower()}"]
    matching = positives if label == "Positive" else negatives
    counter = negatives if label == "Positive" else positives
    tone = texts["tone_positive"] if label == "Positive" else texts["tone_negative"]

    if matching:
        words = ", ".join(f"\"{w}\"" for w in matching)
        base = texts["explanation_with_words"].format(
            label=display_label, tone=tone, words=words
        )
    else:
        base = texts["explanation_no_words"].format(label=display_label)

    if counter and label != "Neutral":
        base += texts["explanation_counter"].format(
            words=", ".join(repr(w) for w in counter),
            label_lower=display_label.lower(),
        )

    base += texts["explanation_confidence"].format(
        strength=strength, conf=f"{confidence:.0%}"
    )
    return base


def transform_text(text: str, target: str, texts: dict) -> str:
    mapping = TO_POSITIVE if target == "Positive" else TO_NEGATIVE
    out = []
    for token in text.split():
        prefix = ""
        suffix = ""
        while token and token[0] in "\"'([":
            prefix += token[0]
            token = token[1:]
        while token and token[-1] in ".,!?;:\"')]":
            suffix = token[-1] + suffix
            token = token[:-1]
        clean = token.lower()
        if clean in mapping:
            replacement = mapping[clean]
            if token[:1].isupper():
                replacement = replacement.capitalize()
            out.append(prefix + replacement + suffix)
        else:
            out.append(prefix + token + suffix)

    rewritten = " ".join(out)
    suffix_key = "transform_suffix_positive" if target == "Positive" else "transform_suffix_negative"
    return rewritten + texts[suffix_key]


def highlight_text(text: str, positives: list[str], negatives: list[str]) -> str:
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
def render_result(result: dict, texts: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric(texts["metric_sentiment"], result["display_label"])
    col2.metric(texts["metric_confidence"], f"{result['confidence']:.2%}")
    col3.metric(texts["metric_stars"], f"{result['stars']} / 5")

    st.progress(result["confidence"])
    render_chart(result["scores"], texts)


def render_chart(scores: dict, texts: dict) -> None:
    order = ["Positive", "Negative"]
    labels_internal = [l for l in order if l in scores]
    display_labels = [texts[f"label_{l.lower()}"] for l in labels_internal]
    values = [scores[l] for l in labels_internal]
    colors = [LABEL_COLORS[l] for l in labels_internal]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(display_labels, values, color=colors, width=0.55)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(texts["chart_ylabel"])
    ax.set_xlabel(texts["chart_xlabel"])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    st.pyplot(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Sentiment Analysis Web App", layout="centered")

    # ── Sprache zuerst wählen — alle weiteren UI-Strings hängen davon ab ──
    if "language" not in st.session_state:
        st.session_state.language = LANGUAGE_OPTIONS[0]
    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    # Beispiel-Button muss VOR dem Textarea geklickt werden, damit der State
    # sich aktualisiert bevor das Widget rendert. Daher hier oben behandeln.
    col_lang, col_example = st.columns([2, 1], vertical_alignment="bottom")
    with col_lang:
        language = st.selectbox(
            "Language",
            options=LANGUAGE_OPTIONS,
            index=LANGUAGE_OPTIONS.index(st.session_state.language),
            key="language",
        )
    if language not in LANGUAGE_OPTIONS:
        st.error("Invalid language. Please choose from the list.")
        st.stop()

    texts = TEXTS[language]

    with col_example:
        if st.button(texts["load_example"], use_container_width=True):
            st.session_state.user_text = texts["example"]

    # ── Titel ────────────────────────────────────────────────
    st.title(texts["title"])
    st.caption(texts["subtitle"])
    st.divider()

    # ── Modell laden ─────────────────────────────────────────
    try:
        classifier = load_model()
    except Exception as exc:  # noqa: BLE001
        st.error(texts["model_error"])
        with st.expander(texts["tech_details"]):
            st.exception(exc)
        st.stop()

    # ── Input ────────────────────────────────────────────────
    st.subheader(texts["input_section"])

    text = st.text_area(
        texts["input_label"],
        key="user_text",
        height=160,
        placeholder=texts["input_placeholder"],
    )

    target_sentiment = st.selectbox(
        texts["target_label"],
        options=TARGET_SENTIMENT_OPTIONS,
        index=0,
        key="target_sentiment",
        help=texts["target_help"],
    )
    if target_sentiment not in TARGET_SENTIMENT_OPTIONS:
        st.error(texts["target_invalid"])
        st.stop()
    st.caption(f"{texts['target_selected']} **{target_sentiment}**")

    has_text = bool(text.strip())
    if not has_text:
        st.caption(f":grey[{texts['no_text_hint']}]")

    analyze_clicked = st.button(
        texts["analyze_button"],
        type="primary",
        use_container_width=True,
        disabled=not has_text,
    )

    st.divider()

    # ── Ergebnis ─────────────────────────────────────────────
    st.subheader(texts["result_section"])

    if not analyze_clicked:
        st.info(texts["result_placeholder"])
        return

    if not has_text:
        st.warning(texts["no_text_warning"])
        return

    with st.spinner(texts["spinner"]):
        try:
            result = analyze_text(classifier, text, texts)
        except Exception as exc:  # noqa: BLE001
            st.error(texts["analysis_error"])
            with st.expander(texts["tech_details"]):
                st.exception(exc)
            return

    st.success(texts["success"].format(
        language=language,
        label=result["display_label"],
        conf=f"{result['confidence']:.0%}",
    ))
    render_result(result, texts)

    # ── Erklärung ────────────────────────────────────────────
    st.divider()
    st.subheader(texts["explanation_section"])

    positives, negatives = find_keywords(text)
    st.markdown(build_explanation(
        result["label"], result["confidence"], positives, negatives, texts
    ))

    if positives or negatives:
        st.markdown(texts["highlighted"])
        st.markdown(highlight_text(text, positives, negatives))

        col_pos, col_neg = st.columns(2)
        col_pos.markdown(
            texts["positive_words_label"] + " "
            + (", ".join(f":green[{w}]" for w in positives) if positives else texts["none"])
        )
        col_neg.markdown(
            texts["negative_words_label"] + " "
            + (", ".join(f":red[{w}]" for w in negatives) if negatives else texts["none"])
        )
    else:
        st.caption(texts["no_keywords_hint"])

    # ── Transformation ───────────────────────────────────────
    st.divider()
    st.subheader(texts["transform_section"])

    tone_key = "transform_tone_positive" if target_sentiment == "Positive" else "transform_tone_negative"
    st.caption(texts["transform_caption"].format(
        target=target_sentiment, tone=texts[tone_key]
    ))
    st.markdown(f"> {transform_text(text, target_sentiment, texts)}")
    st.caption(texts["transform_hint"])


if __name__ == "__main__":
    main()
