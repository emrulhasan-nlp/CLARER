import numpy as np 
import argparse
import os
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("-asp", "--aspects", type=str, default="trip", help="Aspect category (Default: tripadvisor)")
args = parser.parse_args()

model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight transformer model

trip_aspects = ["Service", "Room Size", "Location", "Cleanliness", "Sleep Quality", "Value"]
yelp_aspects = ["Service", "Food Quality", "Ambience", "Cleanliness", "Price", "Menu Variety"]
amz_aspects = ["Plot", "Acting", "Visuals", "Sound Quality", "Genre", "Authenticity"]

if args.aspects == 'trip':
    aspects = trip_aspects
elif args.aspects == 'yelp':
    aspects = yelp_aspects
elif args.aspects == 'amzMT':
    aspects = amz_aspects
else:
    raise ValueError("Invalid aspect category. Choose from: trip, yelp, amz")

def aspect_emb(model, aspects, asp_type):
    aspect_embeddings = model.encode(aspects)
    output_dir = f"../prepadata/{asp_type}/"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{asp_type}_emb.npy"), aspect_embeddings)

aspect_emb(model, aspects, asp_type=args.aspects)
