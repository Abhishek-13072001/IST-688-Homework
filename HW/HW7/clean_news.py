# HW7 - clean the news CSV and remove duplicate articles
import re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
SRC = BASE / "news.csv"
OUT = BASE / "news_clean.csv"


def parse_doc(doc):
    # Document column = title + "Description:" + optional "content:"
    doc = str(doc).strip().strip('"').strip()
    content = ""
    if " content: " in doc:
        doc, content = doc.split(" content: ", 1)
    title, desc = doc, ""
    if " Description: " in doc:
        title, desc = doc.split(" Description: ", 1)
    return title.strip(), desc.strip(), content.strip()


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


# ---- Load ----
df = pd.read_csv(SRC, encoding="utf-8")
df["company_name"] = df["company_name"].str.strip()
print("raw rows:", len(df))

# ---- Split the Document column ----
parsed = df["Document"].apply(parse_doc)
df["title"] = parsed.str[0]
df["description"] = parsed.str[1]
df["content"] = parsed.str[2]
df["date"] = pd.to_datetime(df["Date"], utc=True, format="mixed")
df["source"] = df["URL"].str.extract(r"https?://(?:www\.)?([^/]+)")[0]
df["text"] = (df["title"] + ". " + df["description"] + " " + df["content"]).str.strip()
df["text_len"] = df["text"].str.len()

# ---- Remove duplicates (same company and same title), keep the longest text ----
df["title_key"] = df["title"].apply(norm)
df = (df.sort_values("text_len", ascending=False)
        .drop_duplicates(["company_name", "title_key"])
        .drop_duplicates(["company_name", "URL"])
        .sort_values(["company_name", "date"], ascending=[True, False])
        .reset_index(drop=True))
df.insert(0, "article_id", [f"a{i:04d}" for i in range(len(df))])

# ---- Save ----
keep = ["article_id", "company_name", "date", "source", "URL",
        "title", "description", "content", "text"]
df[keep].rename(columns={"company_name": "company"}).to_csv(OUT, index=False, encoding="utf-8")

print("clean rows:", len(df))
print("companies:", df["company_name"].nunique())
print("empty titles:", (df["title"] == "").sum())
print("rows with content:", (df["content"] != "").sum())
