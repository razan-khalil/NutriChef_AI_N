"""
build_vector_store.py

This script prepares document embeddings and stores them in a ChromaDB vector store.
Run this script ONLY when your dataset changes or you need to rebuild the vector store.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb import PersistentClient
from chromadb.errors import NotFoundError

# === Load environment variables ===
load_dotenv()

# === Load datasets ===
print("📂 Loading recipe datasets...")

recipes_df = pd.read_csv("./database/recipes.csv")
recipes_df["recipe_name"] = recipes_df["recipe_name"].str.lower().str.strip()
recipes_df = recipes_df.rename(columns={"recipe_name": "name"})

raw_df = pd.read_csv("./database/RAW_recipes.csv", usecols=[
    "name", "minutes", "nutrition", "steps", "description", "ingredients"
])
raw_df["name"] = raw_df["name"].str.lower().str.strip()
raw_df["description"] = raw_df["description"].fillna("")
raw_df["steps"] = raw_df["steps"].apply(eval).apply(lambda x: " ".join(x) if isinstance(x, list) else "")
raw_df["ingredients"] = raw_df["ingredients"].apply(eval).apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
raw_df["nutrition"] = raw_df["nutrition"].apply(lambda x: str(x))
raw_df["primaryCategories"] = "general"
raw_df["cuisine_path"] = "unknown"
raw_df["timing"] = raw_df["minutes"].astype(str) + " minutes"

combined_df = pd.concat([recipes_df, raw_df], ignore_index=True)

# === Prepare documents and metadata ===
print("🧾 Preparing documents and metadata...")

docs = combined_df["description"].fillna("") + ". " + combined_df["steps"].fillna("")
metadatas = combined_df[["name", "ingredients", "timing", "cuisine_path"]].to_dict(orient="records")
all_docs = docs.tolist()

# === Generate embeddings ===
print("🧠 Generating embeddings...")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
all_embeddings = embedding_model.embed_documents(all_docs)

# === Setup Chroma vector store ===
print("🗃️ Connecting to Chroma vector store...")

chroma_client = PersistentClient(path="./database/vector_store")

# Delete existing collection (if any)
try:
    chroma_client.delete_collection("recipes")
    print("🗑️ Old 'recipes' collection deleted.")
except NotFoundError:
    print("⚠️ No existing 'recipes' collection found. Skipping deletion.")

collection = chroma_client.get_or_create_collection(name="recipes")

# === Upload in batches ===
def batched_add_to_chroma(collection, docs, embeddings, metadatas, batch_size=5000):
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = [f"recipe_{j}" for j in range(i, i+len(batch_docs))]

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"✅ Added batch {i//batch_size + 1}")

# Run the batch upload
print("📤 Uploading embeddings to Chroma...")
batched_add_to_chroma(collection, all_docs, all_embeddings, metadatas)

print("✅ Vector store build complete!")
