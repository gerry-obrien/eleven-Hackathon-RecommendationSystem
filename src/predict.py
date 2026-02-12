import argparse
import pickle
import os
import pandas as pd
from engine import SegmentRecommender

MODEL_PATH = 'models/'
OUTPUT_PATH = 'output/'

def predict(client_id):
    try:
        with open(os.path.join(MODEL_PATH, 'segment_models.pkl'), 'rb') as f:
            models = pickle.load(f)
        with open(os.path.join(MODEL_PATH, 'client_map.pkl'), 'rb') as f:
            client_map = pickle.load(f)
    except FileNotFoundError:
        print("❌ Models not found! Please run 'python src/train.py' first.")
        return

    client_id = str(client_id)
    if client_id not in client_map:
        print(f"❌ Client ID {client_id} not found.")
        return
        
    meta = client_map[client_id]
    key = (meta['ClientCountry'], meta['ClientSegment'])
    print(f"🔎 Analyzing Client: {client_id} ({key[0]}-{key[1]})")

    if key in models:
        model = models[key]
        # A. Recommendations
        recs = model.recommend(client_id, n_recs=10)
        
        if not recs.empty:
            print("\n✅ TOP RECOMMENDATIONS:")
            print(recs[['ProductID', 'ProductName', 'Stock', 'Score', 'Segment']].to_string(index=False))
            recs.to_csv(os.path.join(OUTPUT_PATH, f'recs_{client_id}.csv'), index=False)
            
            # B. Neighbor Analysis
            print("\n⏳ Generating Neighbor Analysis...")
            neighbors = model.explain_neighbors(client_id, k=5)
            if not neighbors.empty:
                neighbors.to_csv(os.path.join(OUTPUT_PATH, f'neighbors_{client_id}.csv'), index=False)
                print(f"📄 Saved detailed analysis files to {OUTPUT_PATH}")
        else:
            print("⚠️ No recommendations generated.")
    else:
        print(f"⚠️ No model for segment: {key}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('client_id', type=str)
    args = parser.parse_args()
    predict(args.