import os 
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR.parent / 'images'
OUTPUT_PATH = BASE_DIR.parent / 'data' / 'image_embeddings.pkl'

# Load image embedding model
model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/5"
model = hub.load(model_url)

# Process image with TensorFlow
def process_image(image_path):
    image = Image.open(image_path)
    if image.mode == "P" or image.mode == "RGBA":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0 # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return tf.convert_to_tensor(image, dtype=tf.float32)

# Extract Embeddings
embeddings, filename = [], []
image_files = list(IMAGE_DIR.glob('*.jpg')) + \
                list(IMAGE_DIR.glob('*.png')) + \
                    list(IMAGE_DIR.glob('*.jpeg')) + \
                        list(IMAGE_DIR.glob('*.webp'))
print(f"Found {len(image_files)} image files in 'images/'")

for img_path in tqdm(image_files, desc="Embedding images:"):
    # img_name = img_path.name
    try:
        image_tensor = process_image(img_path)
        embedding = model(image_tensor).numpy().flatten()
        embeddings.append(embedding)
        filename.append(img_path.name)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Save as Pickle file
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({"filenames": filename, "embeddings": embeddings}, f)

print(f"Saved {len(embeddings)} image embeddings to {OUTPUT_PATH}")