import pickle
import pandas as pd
import json
import csv
from typing import List

# =============================== #
#           Config                #
# =============================== #

AMAZON_PICKLE_PATH = "../rawdata/amzMT/reviews.pickle"
JSON_INPUT_PATH = '../rawdata/amzMT/Movies_and_TV.json'
JSON_TO_CSV_PATH = '../rawdata/amzMT/Movies_and_TV.csv'
CLEANED_PICKLE_CSV_PATH = '../rawdata/amzMT/amz_revdf.csv'
FINAL_OUTPUT_PATH = '../prepdata/AmzMT/amz_mt.csv'


# =============================== #
#         Utility Functions       #
# =============================== #

def load_pickle_dataframe(filepath: str) -> pd.DataFrame:
    """Load a pickle file and return a DataFrame."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
    df.rename(columns={'asin': 'item'}, inplace=True)
    return df


def convert_json_to_csv(json_path: str, csv_path: str):
    """Convert a JSON lines file to a structured CSV."""
    with open(json_path, 'r', encoding='utf-8') as json_file, \
         open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(['reviewerID', 'asin', 'reviewText', 'overall'])

        for line in json_file:
            try:
                data = json.loads(line)
                writer.writerow([
                    data.get('reviewerID', ''),
                    data.get('asin', ''),
                    data.get('reviewText', ''),
                    data.get('overall', '')
                ])
            except json.JSONDecodeError as e:
                print(f"[Warning] Skipping invalid JSON line: {e}")
    print(f"[Success] JSON converted to CSV: {csv_path}")


def load_and_prepare_raw_csv(filepath: str) -> pd.DataFrame:
    """Load raw CSV, rename fields, and remove duplicates."""
    df = pd.read_csv(filepath)
    df.rename(columns={'asin': 'item', 'reviewerID': 'user', 'reviewText': 'text'}, inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def extract_explanations(template_column: List) -> List[str]:
    """Extract the explanation from the template field."""
    explanations = []
    for item in template_column:
        try:
            _, _, exp, _ = item
        except ValueError:
            try:
                item_tuple = eval(item)
                _, _, exp, _ = item_tuple
            except (ValueError, SyntaxError, TypeError):
                print(f"[Warning] Skipping invalid template: {item}")
                continue
        explanations.append(exp)
    return explanations


def merge_and_clean_data(pickle_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Merge dataframes and create a clean, final dataset."""
    merged = pd.merge(pickle_df, raw_df[['user', 'item', 'text']], on=['user', 'item'], how='left')
    merged.drop_duplicates(subset=['user', 'item'], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Drop unnecessary columns
    merged.drop(columns=['predicted', 'overall'], errors='ignore', inplace=True)

    # Extract explanations
    merged['explanation'] = extract_explanations(merged['template'])
    merged.drop(columns=['template'], inplace=True)

    # Select and reorder columns
    final_df = merged[['user', 'item', 'text', 'explanation', 'rating']]
    return final_df


# =============================== #
#              Main               #
# =============================== #

if __name__ == "__main__":
    # Step 1: Load pickle data and save intermediate CSV
    amz_df = load_pickle_dataframe(AMAZON_PICKLE_PATH)
    amz_df.to_csv(CLEANED_PICKLE_CSV_PATH, index=False)

    # Step 2: Convert JSON reviews to CSV
    convert_json_to_csv(JSON_INPUT_PATH, JSON_TO_CSV_PATH)

    # Step 3: Load and prepare the raw Amazon review CSV
    raw_df = load_and_prepare_raw_csv(JSON_TO_CSV_PATH)

    # Step 4: Merge datasets and clean
    final_df = merge_and_clean_data(amz_df, raw_df)

    # Step 5: Save final output
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)
    print(f"[Success] Final dataset saved: {FINAL_OUTPUT_PATH}")
