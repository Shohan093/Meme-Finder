import os
import re
import json
import pickle
import time

from pathlib import Path
from dotenv import load_dotenv
from google.generativeai import configure, embed_content
from tqdm import tqdm

import google.generativeai as genai

# File paths
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR.parent / 'data'
input_json = data_dir / 'meme_texts.json'
clean_json = data_dir / 'clean_text.json'
output_pkl = data_dir / 'text_embeddings.pkl'

# Load environment
load_dotenv(BASE_DIR.parent / '.env')
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")

configure(api_key=api_key)

# Load meme texts from JSON file
with open(input_json, 'r', encoding='utf-8') as f:
    meme_data = json.load(f)

# Clean the text data
def clean_text(text):
    text = text.encode("ascii", "ignore").decode()  # Remove non-ASCII
    text = re.sub(r'https?://\S+', '', text)        # remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()        # remove extra whitespace
    return text

clean_data = []

for filename, text in meme_data.items():
    cleaned_text = clean_text(text)
    if len(cleaned_text) >= 5: # Discard very short texts
        clean_data.append({
            "filename": filename,
            "text": cleaned_text
        })

# Save cleaned data to JSON
with open(clean_json, 'w', encoding='utf-8') as f:
    json.dump(clean_data, f, indent=4, ensure_ascii=False)

# Embed and collect image texts
emebedded_memes = []

# Load cleaned data
with open(clean_json, 'r', encoding='utf-8') as f:
    clean_data = json.load(f)

print(f"Embedding {len(clean_data)} meme texts...")

# Retry embedding if it fails
def embed_with_retry(text, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return response["embedding"]
        except Exception as e:
            print(f"Attempt {attempt + 1} / {retries} failed: {e}")
            time.sleep(delay)
    raise Exception("All attempts to embed text failed.")


for idx, item in enumerate(tqdm(clean_data, desc='Embedding Texts')):
    text = item["text"].strip()
    file_name = item["filename"]

    if not text:
        continue
    embedding = embed_with_retry(text) # Getting Embedding
    if embedding:    
        emebedded_memes.append({
            "filename": file_name,
            "text": text,
            "embedding": embedding
        })
    if idx and idx % 1000 == 0:
        print(f"\nEmbedded {idx} / {len(clean_data)}")

    time.sleep(0.5) # To avoid hitting rate limits

# Save to pickle file
with open(output_pkl, 'wb') as f:
    pickle.dump(emebedded_memes, f)

print(f"Saved embeddings to {output_pkl}")