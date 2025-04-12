import os
import pickle
import time
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from prompt_rewriter import rewrite_promp_with_gemini

import google.generativeai as genai

# Path
BASE_DIR = Path(__file__).resolve().parent
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

# Normalize score
def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) # Avoid division by zero

# Hybrid search function
def hybrid_search(user_input: str):
    rewritten_prompt = rewrite_promp_with_gemini(user_input)
    
    # Embed query
    query_embedding = embed_query_text(rewritten_prompt)

    # Prepare lists
    text_scores, image_score = [], []
    all_filenames = []

    for item in combined_data:
        text_embed = np.array(item["text_embedding"])
        image_embed = np.array(item["image_embedding"])
        filename = item["filename"]

        # Similarities
        sim_text = cosine_similarity(query_embedding.reshape(1, -1), text_embed.reshape(1, -1)).flatten()[0]
        sim_image = cosine_similarity(query_embedding.reshape(1, -1), image_embed.reshape(1, -1)).flatten()[0]

        text_scores.append(sim_text)
        image_score.append(sim_image)
        all_filenames.append(filename)

    # Normalize scores
    norm_text_scores = normalize(text_scores)
    norm_image_scores = normalize(image_score)

    final_scores = (TEXT_WEIGHT * norm_text_scores) + (IMAGE_WEIGHT * norm_image_scores)

    # Rank
    ranked_indices = np.argsort(final_scores)[::-1][:TOP_K]
    top_indeces = ranked_indices[:TOP_K]

    print(f"\nTop matches\n")
    for i in top_indeces:
        print(f"{i + 1}. Filename: {all_filenames[i]}")
        print(f"\tText Similarity: {text_scores[i]:.4f}")
        print(f"\nImage Similarity: {image_score[i]:.4f}")
        print(f"\tCombined Score: {final_scores[i]:.4f}\n")
        


if __name__ == "__main__":
    user_input = input("Enter your search prompt: ")
    hybrid_search(user_input)