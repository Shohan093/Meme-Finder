import os
import pickle
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import google.generativeai as genai
from google.generativeai import embed_content

from prompt_rewriter import rewrite_promp_with_gemini

# Paths
BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_FILE = BASE_DIR.parent / 'data' / 'text_embeddings.pkl'
IMAGE_DIR = BASE_DIR.parent / 'images'

# Load environment variables
load_dotenv(BASE_DIR.parent / '.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)

# Load embedded memes
with open(EMBEDDING_FILE, 'rb') as f:
    embedded_memes = pickle.load(f)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def normalize(v):
    return v / np.linalg.norm(v)

raw_input = input("Enter your search prompt: ")
query = rewrite_promp_with_gemini(raw_input)

response = embed_content(
    model="models/embedding-001",
    content=query,
    task_type="retrieval_document"
)

query_embedding = normalize(np.array(response["embedding"]))

# Find top-K similar memes
top_k = 7
# top_results = sorted(embedded_memes, key=lambda x: cosine_similarity(query_embedding, np.array(x["embedding"])), reverse=True)[:top_k]
similarities = []

for meme in embedded_memes:
    meme_vector = normalize(np.array(meme["embedding"]))
    sim = np.dot(query_embedding, meme_vector)
    similarities.append((sim, meme))

# Sort by similarity
similarities.sort(reverse=True, key=lambda x: x[0])
top_results = similarities[:top_k]

# Display results
print("\nTop Matches\n")
plt.figure(figsize=(12, 6))
for i, (score, meme) in enumerate(top_results):
    print(f"{i + 1}. Filename: {meme['filename']}")
    # print(f"\tSimilarity Score: {score:.4f}")
    print(f"\tSimilarity Text: {meme['text']}\n")

    # Show image
    if os.path.exists(IMAGE_DIR / meme["filename"]):
        image_path = IMAGE_DIR / meme["filename"]
        image = Image.open(image_path)
        
        plt.subplot(1, top_k, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"{meme['filename'][:15]}...", fontsize=8)
    else:
        print(f"Image {meme['filename']} could not be found.\n")

plt.tight_layout()
plt.show()