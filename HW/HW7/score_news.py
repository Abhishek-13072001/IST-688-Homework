# HW7 - score every article for how interesting it is to a law firm
# usage: python HW/HW7/score_news.py gpt-5-mini
#        python HW/HW7/score_news.py gpt-5 --sample 100
import json
import os
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from openai import OpenAI

BASE = Path(__file__).parent
CSV_PATH = BASE / "news_clean.csv"
DB_PATH = BASE / "HW7_chroma_db"
COLLECTION = "HW7News"
BATCH = 5          # articles per API call
WORKERS = 6        # calls running at the same time
SIM_CUTOFF = 0.85  # similarity needed to count as "same story"

SYSTEM_PROMPT = """You help a large global law firm monitor news about its clients.
For each article, rate how interesting it is to the firm's lawyers.

relevance (0-10):
9-10 = lawsuit, regulatory action, investigation, fine, merger or acquisition, major compliance or antitrust issue
6-8 = leadership change, big contract, legal risk, major financial event
3-5 = general business news with a possible legal angle
0-2 = routine news, stock tips, marketing, no legal angle

about_company: false if the article is not really about the named company
(for example the name matches something else).

reason: one short sentence on why a lawyer would care.

Return JSON only, in this format:
{"results": [{"id": "a0001", "relevance": 7, "about_company": true, "reason": "..."}]}"""


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


def score_batch(client, model, rows):
    # one API call for a small group of articles
    lines = []
    for r in rows:
        lines.append(f"id: {r['article_id']}\ncompany: {r['company']}\nsource: {r['source']}\n"
                     f"text: {r['text'][:1200]}\n")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": "\n---\n".join(lines)}],
            response_format={"type": "json_object"},
            reasoning_effort="low",
        )
        return json.loads(resp.choices[0].message.content)["results"]
    except Exception as e:
        print("batch failed:", str(e)[:100])
        return []


def get_coverage():
    # how many other outlets ran a very similar story for the same company
    collection = chromadb.PersistentClient(path=str(DB_PATH)).get_collection(COLLECTION)
    data = collection.get(include=["embeddings", "metadatas"])
    emb = np.array(data["embeddings"])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    companies = np.array([m["company"] for m in data["metadatas"]])
    sources = np.array([m["source"] for m in data["metadatas"]])
    ids = np.array(data["ids"])

    coverage = {}
    for c in np.unique(companies):
        idx = np.where(companies == c)[0]
        sim = emb[idx] @ emb[idx].T
        same_story = (sim >= SIM_CUTOFF) & (sources[idx][:, None] != sources[idx][None, :])
        for i, n in zip(idx, same_story.sum(axis=1)):
            coverage[ids[i]] = int(n)
    return coverage


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-5-mini"
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
    out_path = BASE / f"scores_{model}.csv"

    client = OpenAI(api_key=get_api_key())
    df = pd.read_csv(CSV_PATH, encoding="utf-8").fillna("")
    if sample:
        df = df.sample(sample, random_state=42)

    # skip articles that already have a score
    done = pd.DataFrame(columns=["article_id", "relevance", "about_company", "reason"])
    if out_path.exists():
        done = pd.read_csv(out_path)[["article_id", "relevance", "about_company", "reason"]]
    todo = df[~df["article_id"].isin(done["article_id"])]
    print(f"model: {model} | to score: {len(todo)} | already done: {len(done)}")

    rows = todo.to_dict("records")
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda b: score_batch(client, model, b), batches))

    new = [r for batch in results for r in batch]
    new_df = pd.DataFrame(new).rename(columns={"id": "article_id"})
    scores = pd.concat([done, new_df], ignore_index=True).drop_duplicates("article_id")
    scores = scores[scores["article_id"].isin(df["article_id"])]

    # final score out of 100
    scores["relevance"] = pd.to_numeric(scores["relevance"], errors="coerce").fillna(0)
    scores.loc[scores["about_company"].astype(str).str.lower() == "false", "relevance"] = 0
    coverage = get_coverage()
    scores["coverage"] = scores["article_id"].map(coverage).fillna(0).astype(int)
    scores["score"] = scores["relevance"] * 9 + scores["coverage"].clip(upper=5) * 2

    scores.to_csv(out_path, index=False, encoding="utf-8")
    missing = len(df) - len(scores)
    print(f"saved {len(scores)} scores to {out_path.name}")
    if missing:
        print(f"{missing} articles failed, run the same command again to retry them")
    print(scores.sort_values("score", ascending=False).head(5)[["article_id", "score", "reason"]])


if __name__ == "__main__":
    main()
