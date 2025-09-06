import pandas as pd
import re
import emoji
import spacy

# Cargar modelo SpaCy una sola vez
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # desactiva lo que no uses

def preprocess_text_batch(texts: list,
                          lemmatize: bool = True,
                          remove_stopwords: bool = True,
                          handle_emojis: str = 'demojize',
                          remove_punctuation: bool = True,
                          normalize_lengthening: bool = True):
    """
    Procesa una lista de textos aplicando limpieza, normalización de alargamientos,
    manejo de emojis, lematización y eliminación de stopwords/puntuación.
    """
    processed_texts_for_spacy = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'http\S+|@\w+', '', text)  # eliminar URLs y menciones
        if normalize_lengthening:
            text = re.sub(r'(.)\1{2,}', r'\1', text)
        if handle_emojis == 'demojize':
            text = emoji.demojize(text, delimiters=(" ", " "))
        elif handle_emojis == 'drop':
            text = emoji.demojize(text, delimiters=("", ""))
            text = re.sub(r':\S+:', '', text)
        processed_texts_for_spacy.append(text)

    final_processed_texts = []
    for doc in nlp.pipe(processed_texts_for_spacy, n_process=-1, batch_size=1000):
        processed_tokens = []
        for token in doc:
            if remove_punctuation and token.is_punct:
                continue
            if remove_stopwords and token.is_stop:
                continue
            token_text = token.lemma_ if lemmatize else token.text
            if token_text.strip():
                processed_tokens.append(token_text)
        final_processed_texts.append(" ".join(processed_tokens))

    return final_processed_texts

# === PIPELINE PRINCIPAL ===
if __name__ == "__main__":
    # 1. Cargar el dataset
    column_names = ["sentiment", "id", "date", "query", "user", "text"]
    df = pd.read_csv("sentiment140_data/training.1600000.processed.noemoticon.csv",encoding='ISO-8859-1',names=column_names)  # o pd.read_parquet("dataset.parquet")

    # 2. Filtrar y limpiar columnas (como en tu código original)
    df = df[df["sentiment"] != 2].copy()
    df["label"] = df["sentiment"].apply(lambda x: 1 if x == 4 else 0)
    df.drop(columns=["sentiment", "id", "date", "query", "user"], inplace=True)

    # 3. Procesar la columna de texto (en batch, usando nlp.pipe)
    texts = df["text"].tolist()
    processed_texts = preprocess_text_batch(
        texts,
        lemmatize=False,
        remove_stopwords=False,
        remove_punctuation=True
    )

    df["text"] = processed_texts

    # 4. Guardar resultado
    df.to_parquet("dataset_procesado_remove_punctuation_true.parquet", index=False)

