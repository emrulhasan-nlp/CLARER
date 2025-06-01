import pickle
import pandas as pd
import json
import csv
from typing import List


# =============================== #
#           Config                #
# =============================== #

YELP_PICKLE_PATH = "../rawdata/yelp/reviews.pickle"
YELP_JSON_PATH = "../rawdata/yelp/yelp_academic_dataset_review.json"
YELP_CSV_PATH = "../rawdata/yelp/yelp_review.csv"
YELP_FINAL_OUTPUT = "../prepdata/yelp/yelp.csv"


# =============================== #
#         Utility Functions       #
# =============================== #

def load_pickle_dataframe(path: str) -> pd.DataFrame:
    """Load a pickle file and return a cleaned DataFrame with extracted explanations."""
    with open(path, 'rb') as f:
        reviews = pickle.load(f)

    df = pd.DataFrame(reviews)
    df.drop(columns=['predicted'], errors='ignore', inplace=True)

    # Extract explanation from template
    explanations = []
    for item in df['template']:
        try:
            _, _, exp, _ = item
            explanations.append(exp)
        except (ValueError, TypeError):
            explanations.append("")
            print(f"[Warning] Skipping invalid template: {item}")
    df['explanation'] = explanations

    df.drop(columns=['template'], inplace=True)
    return df


def convert_json_to_csv(json_input_path: str, csv_output_path: str):
    """Convert Yelp JSON lines to structured CSV."""
    with open(json_input_path, 'r', encoding='utf-8') as json_file, \
         open(csv_output_path, 'w', newline='', encoding='utf-8') as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(['user_id', 'business_id', 'text', 'stars'])

        for line in json_file:
            try:
                data = json.loads(line)
                writer.writerow([
                    data.get('user_id', ''),
                    data.get('business_id', ''),
                    data.get('text', ''),
                    data.get('stars', '')
                ])
            except json.JSONDecodeError as e:
                print(f"[Warning] Skipping invalid JSON line: {e}")
    print(f"[Success] JSON converted to CSV: {csv_output_path}")


def load_and_prepare_yelp_csv(csv_path: str) -> pd.DataFrame:
    """Load and prepare Yelp CSV file."""
    df = pd.read_csv(csv_path)
    df.rename(columns={
        'user_id': 'user',
        'business_id': 'item',
        'stars': 'rating'
    }, inplace=True)
    return df


# =============================== #
#              Main               #
# =============================== #

if __name__ == "__main__":
    # Step 1: Load explanation-enhanced reviews
    yelp_df = load_pickle_dataframe(YELP_PICKLE_PATH)

    # Step 2: Convert JSON reviews to CSV if needed
    convert_json_to_csv(YELP_JSON_PATH, YELP_CSV_PATH)

    # Step 3: Load original Yelp review CSV
    review_df = load_and_prepare_yelp_csv(YELP_CSV_PATH)

    # Step 4: Merge and save final dataset
    final_df = pd.merge(yelp_df, review_df, on=['user', 'item'], how='inner')
    final_df.to_csv(YELP_FINAL_OUTPUT, index=False)
    print(f"[Success] Final Yelp dataset saved: {YELP_FINAL_OUTPUT}")
