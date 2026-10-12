# HW7 - build the ChromaDB from news_clean.csv (run this once, then commit HW7_chroma_db)
import os
import tomllib
from pathlib import Path

import chromadb
import pandas as pd
from openai import OpenAI

BASE = Path(__file__).parent
CSV_PATH = BASE / "news_clean.csv"
DB_PATH = BASE / "HW7_chroma_db"
COLLECTION = "HW7News"
EMBED_MODEL = "text-embedding-3-small"
BATCH = 100


def get_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    secrets = BASE.parent.parent / ".streamlit" / "secrets.toml"
    if secrets.exists():
        with open(secrets, "rb") as f:
            key = tomllib.load(f).get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not found")
    return key


client = OpenAI(api_key=get_api_key())

# ---- Load articles ----
df = pd.read_csv(CSV_PATH, encoding="utf-8").fillna("")
expected = len(df)
print("Articles to embed:", expected)

# ---- Open the collection ----
chroma = chromadb.PersistentClient(path=str(DB_PATH))
collection = chroma.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

# compare with the exact expected count, not just count() > 0
if collection.count() == expected:
    print("DB already complete, nothing to do.")
else:
    if collection.count() > 0:
        print(f"Partial DB ({collection.count()}/{expected}), rebuilding...")
        chroma.delete_collection(COLLECTION)
        collection = chroma.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

    # ---- Embed in batches ----
    for start in range(0, expected, BATCH):
        batch = df.iloc[start:start + BATCH]
        docs = [t[:6000] for t in batch["text"]]  # cut very long articles
        resp = client.embeddings.create(model=EMBED_MODEL, input=docs)
        collection.add(
            ids=batch["article_id"].tolist(),
            documents=docs,
            embeddings=[d.embedding for d in resp.data],
            metadatas=[
                {
                    "article_id": r.article_id,
                    "company": r.company,
                    "date": str(r.date)[:10],
                    "source": r.source,
                    "url": r.URL,
                    "title": r.title,
                }
                for r in batch.itertuples()
            ],
        )
        print(f"  embedded {min(start + BATCH, expected)}/{expected}")

# ---- Check the result ----
final = collection.count()
assert final == expected, f"Expected {expected} but DB has {final}"
print("Done. DB has", final, "items")

# quick test search
q = "lawsuit or regulatory investigation"
emb = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding
res = collection.query(query_embeddings=[emb], n_results=3)
print("Test search:", q)
for meta in res["metadatas"][0]:
    print(f"  - [{meta['company']}] {meta['title'][:90]}")
