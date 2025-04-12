import os
import pickle

from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Path
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'data'
TEXT_EMBED_PATH = DATA_DIR / 'text_embeddings.pkl'
IMAGE_EMBED_PATH = DATA_DIR / 'image_embeddings.pkl'
OUTPUT_PATH = DATA_DIR / 'combined_embeddings.pkl'

# Load embeddigns
with open(TEXT_EMBED_PATH, 'rb') as f:
    text_embeddings = pickle.load(f)

with open(IMAGE_EMBED_PATH, 'rb') as f:
    image_embeddings = pickle.load(f)

# Convert image embeddings to a dictionary for easy access
image_dict = {
    fname: emb for fname, emb in zip(image_embeddings["filenames"], image_embeddings["embeddings"])
}

# Combine base on filename
combined_embeddings = []

print("Combining text and image embeddings...")
for text_item in tqdm(text_embeddings, desc="Combining"):
    filename = text_item["filename"]
    if filename in image_dict:
        combined_embeddings.append({
            "filename": filename,
            "text": text_item["text"],
            "text_embedding": text_item["embedding"],
            "image_embedding": image_dict[filename]
        })
    else:
        print(f"Warning: {filename} not found in image embeddings. Skipping.")

# Save combined embeddings to a file
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(combined_embeddings, f)

print(f"Combined {len(combined_embeddings)} embeddings saved to {OUTPUT_PATH.relative_to(BASE_DIR.parent)}")