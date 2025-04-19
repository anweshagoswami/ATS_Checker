# train_model.py

import argparse
import pickle
from ats_model import process_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Train the ATS keyword model and save it to disk."
    )
    parser.add_argument(
        "--csv", "-c",
        required=True,
        help="Path to CSV dataset (must contain 'JOB TITLE' & 'RESUME TEXT')."
    )
    parser.add_argument(
        "--top_n", "-n",
        type=int,
        default=10,
        help="Number of top TF‑IDF keywords per job title."
    )
    parser.add_argument(
        "--output", "-o",
        default="learned_keywords.pkl",
        help="File path for the pickled learned‑keywords dict."
    )
    args = parser.parse_args()

    # Learn keywords (this may take a moment)
    _, learned_keywords = process_dataset(args.csv, top_n_keywords=args.top_n)

    # Save the dict so you never retrain at inference time
    with open(args.output, "wb") as f:
        pickle.dump(learned_keywords, f)

    print(f"✅ Learned keywords saved to '{args.output}'.")

if __name__ == "__main__":
    main()
