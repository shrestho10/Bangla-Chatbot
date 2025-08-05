import pandas as pd
import re
import math

# === Load Dataset ===
DATA_PATH = "new_paper_data.csv"
df = pd.read_csv(DATA_PATH)
df.dropna(subset=['text'], inplace=True)
df.drop_duplicates(subset=['text'], inplace=True)

# === Bangla Stopwords (you can add more manually) ===
bangla_stopwords = set([
    "ও", "এবং", "আর", "তবে", "যদি", "কিন্তু", "এই", "উপর", "নিয়ে", "একটি", "সব", "যা", "করা", "হয়",
    "ছিল", "যার", "তাদের", "তার", "তাদেরকে", "আপনি", "আমি", "তুমি", "সে", "এটি", "ছিলো"
])

# === Preprocessing ===
def clean_and_tokenize(text):
    # Keep only Bangla characters and space
    text = re.sub(r"[^\u0980-\u09FF\s]", " ", text)
    tokens = text.strip().split()
    tokens = [word for word in tokens if word not in bangla_stopwords]
    return tokens

# Apply preprocessing
df['tokens'] = df['text'].apply(clean_and_tokenize)

# === Create Vocabulary ===
def build_vocabulary(token_lists):
    vocab = set()
    for tokens in token_lists:
        vocab.update(tokens)
    return sorted(vocab)

vocab = build_vocabulary(df['tokens'])
word_to_index = {word: idx for idx, word in enumerate(vocab)}
N = len(df)

# === Compute IDF ===
def compute_idf():
    df_counts = [0] * len(vocab)
    for tokens in df['tokens']:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in word_to_index:
                df_counts[word_to_index[token]] += 1

    idf = [0] * len(vocab)
    for i, df_count in enumerate(df_counts):
        idf[i] = math.log((N + 1) / (df_count + 1)) + 1  # smooth
    return idf

idf_vector = compute_idf()

# === Compute TF-IDF for all sentences ===
def compute_tf(tokens):
    tf = [0] * len(vocab)
    for token in tokens:
        if token in word_to_index:
            idx = word_to_index[token]
            tf[idx] += 1
    token_count = len(tokens)
    return [count / token_count if token_count > 0 else 0 for count in tf]

def compute_tfidf(tf, idf):
    return [tf[i] * idf[i] for i in range(len(tf))]

# Compute all tf-idf vectors
df['tf'] = df['tokens'].apply(compute_tf)
df['tfidf'] = df['tf'].apply(lambda tf: compute_tfidf(tf, idf_vector))

# === Cosine Similarity ===
def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# === Chatbot Logic ===
def respond_to_query(query, top_n=2):
    query_tokens = clean_and_tokenize(query)
    query_tf = compute_tf(query_tokens)
    query_tfidf = compute_tfidf(query_tf, idf_vector)

    similarities = []
    for i, row in df.iterrows():
        sim = cosine_similarity(query_tfidf, row['tfidf'])
        similarities.append((sim, row['text']))

    similarities.sort(reverse=True, key=lambda x: x[0])
    top_responses = [text for _, text in similarities[:top_n]]
    return top_responses

# === Command Line Chat Interface ===
def run_chatbot():
    print("📣 Bangla Chatbot (TF-IDF & Cosine Similarity, No Libraries)")
    print("➡️ Type 'exit' to quit.\n")
    while True:
        user_input = input("🧑‍💻 You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        responses = respond_to_query(user_input)
        for i, res in enumerate(responses, 1):
            print(f"🤖 Response {i}: {res}")

# === Run it ===
if __name__ == '__main__':
    run_chatbot()
