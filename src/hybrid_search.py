import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from prompt_rewriter import rewrite_promp_with_gemini

import google.generativeai as genai

# Path
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR.parent / 'images'
DATA_DIR = BASE_DIR.parent / 'data'
EMBEDDING_FILE = DATA_DIR / 'combined_embeddings.pkl'

# Load environment variables
load_dotenv(BASE_DIR.parent / '.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)

# Load combined embeddings
with open(EMBEDDING_FILE, 'rb') as f:
    combined_data = pickle.load(f)

# MLP for mapping 1792 -> 768
class EmbeddingMapper(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(768)

    def call(self, X):
        X = self.dense1(X)
        return self.dense2(X)
    
# Instantiate the model
mapper_model = EmbeddingMapper()

# Parameters
TEXT_WEIGHT = 0.6
IMAGE_WEIGHT = 0.4
TOP_K = 10

# Embed user query
def embed_query_text(text, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return np.array(response["embedding"])
        except Exception as e:
            print(f"Error embedding text: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception("Failed to embed text after multiple attempts.")

def plot_results(results):
    n = len(results)
    cols = 2
    rows = (n + 1) // cols

    plt.figure(figsize=(12, 5 * rows))
    for i, result in enumerate(results):
        img_path = IMAGE_DIR / result["filename"]
        try:
            img = Image.open(img_path)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Filename: {result['filename']}\nText Similarity: {result['text_similarity']:.4f}\nImage Similarity: {result['image_similarity']:.4f}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    plt.tight_layout()
    plt.show()

# Normalize score
def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) # Avoid division by zero

# Hybrid search function
def hybrid_search(user_input: str):
    rewritten_prompt = rewrite_promp_with_gemini(user_input)
    
    # Embed query
    query_embedding = normalize(np.array(embed_query_text(rewritten_prompt)))

    results = []
    for entry in combined_data:
        text_embed = normalize(np.array(entry["text_embedding"]))
        image_embed = normalize(np.array(entry["image_embedding"]))
        image_embed = mapper_model(np.expand_dims(image_embed, axis=0)).numpy().flatten()
        image_embed = normalize(image_embed)

        sim_text = cosine_similarity(query_embedding.reshape(1, -1), text_embed.reshape(1, -1))[0]
        sim_image = cosine_similarity(query_embedding.reshape(1, -1), image_embed.reshape(1, -1))[0]
        hybrid_score = TEXT_WEIGHT * sim_text + IMAGE_WEIGHT * sim_image

        results.append({
            "filename": entry["filename"],
            "text": entry["text"],
            "hybrid_score": hybrid_score
        })
        
        # Store similarity scores for plotting
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_results = results[:TOP_K]

        plot_results(top_results)


if __name__ == "__main__":
    user_input = input("Enter your search prompt: ")
    hybrid_search(user_input)