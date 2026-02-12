# reco_refacto.py
from pathlib import Path
import argparse
import json
import joblib

from src.newby_reco.data import load_reco_data
from src.newby_reco.pipeline import make_preprocess_pipeline
from src.newby_reco.model import (
    fit_knn_model,
    recommend_for_client_id,
    recommend_for_profile
)

MODEL_PATH = "knn_reco_bundle.pkl"


def train_and_save(raw_dir: Path):
    reco_data = load_reco_data(raw_dir)
    preprocessor = make_preprocess_pipeline()

    model = fit_knn_model(
        existing_clients=reco_data.existing_clients,
        preprocessor=preprocessor,
        client_products=reco_data.client_products,
        client_country=reco_data.client_country,
        stock_map=reco_data.stock_map,
        products=reco_data.products,
        n_neighbors_fit=50,
    )

    joblib.dump(model, MODEL_PATH)
    print(f"[OK] Model saved to {MODEL_PATH}")
    return model, reco_data


def load_model():
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Loaded model from {MODEL_PATH}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--train", action="store_true", help="Train model and save bundle")
    parser.add_argument("--client_id", type=int, default=None, help="Recommend for a client ID (existing or newby)")
    parser.add_argument("--profile_json", type=str, default=None, help="Path to a JSON profile for anonymous prediction")
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--k_reco", type=int, default=10)
    parser.add_argument("--out_csv", type=str, default=None, help="Optional output CSV path for reco df")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    if args.train:
        model, reco_data = train_and_save(raw_dir)
    else:
        model = load_model()
        # we still need clients/products/stocks for inference (clients table for client_id mode)
        reco_data = load_reco_data(raw_dir)

    if args.client_id is not None:
        reco_df, neighbors_df = recommend_for_client_id(
            model=model,
            clients_df=reco_data.clients,
            query_client_id=args.client_id,
            k_neighbors=args.k_neighbors,
            k_reco=args.k_reco
        )
        print("\n--- NEIGHBORS ---")
        print(neighbors_df.head(10).to_string(index=False))
        print("\n--- RECOMMENDATIONS ---")
        print(reco_df.head(50).to_string(index=False))

        if args.out_csv:
            reco_df.to_csv(args.out_csv, index=False)
            print(f"[OK] Saved recommendations to {args.out_csv}")

    elif args.profile_json is not None:
        with open(args.profile_json, "r") as f:
            profile = json.load(f)

        reco_df, neighbors_df = recommend_for_profile(
            model=model,
            profile=profile,
            k_neighbors=args.k_neighbors,
            k_reco=args.k_reco
        )
        print("\n--- NEIGHBORS ---")
        print(neighbors_df.head(10).to_string(index=False))
        print("\n--- RECOMMENDATIONS ---")
        print(reco_df.head(50).to_string(index=False))

        if args.out_csv:
            reco_df.to_csv(args.out_csv, index=False)
            print(f"[OK] Saved recommendations to {args.out_csv}")

    else:
        print("Nothing to do. Use --client_id <ID> or --profile_json <path> (and optionally --train).")


if __name__ == "__main__":
    main()
