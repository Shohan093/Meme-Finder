import os
import json
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

