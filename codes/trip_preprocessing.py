import json
import string
import pandas as pd
import csv
import pickle
from typing import Dict, List

# =============================== #
#           Config                #
# =============================== #

PREDEFINED_SUBRATINGS = ['Value', 'Location', 'Sleep Quality', 'Rooms', 'Cleanliness', 'Service']
REVIEW_JSON_PATH = '../rawdata/trip/OriginalReviews.json'
RAW_CSV_PATH = '../rawdata/trip/rawreview.csv'
EXPLANATION_PICKLE_PATH = '../rawdata/trip/reviews.pickle'
EXPLANATION_CSV_PATH = '../rawdata/trip/tripdata.csv'
MERGED_CSV_PATH = '../prepdata/trip/trip_review_xplns.csv'


# =============================== #
#         Utility Functions       #
# =============================== #

def is_english_simple(text: str) -> bool:
    """Check if the text contains only English letters and common punctuation."""
    allowed_chars = set(string.ascii_letters + string.whitespace + string.punctuation)
    return all(char in allowed_chars for char in text)


def align_subratings(review: Dict, predefined_keys: List[str]) -> Dict:
    """Ensure all predefined subratings are present in each review."""
    subratings = review.get('subRatings', {})
    aligned = {key: subratings.get(key, 0) for key in predefined_keys}
    review['subRatings'] = aligned
    return review


# =============================== #
#       Data Processing           #
# =============================== #

def load_and_process_reviews(filepath: str) -> List[Dict]:
    """Load reviews from JSON and align subratings."""
    processed_reviews = []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

            for entry in data:
                review = align_subratings(entry, PREDEFINED_SUBRATINGS)
                processed_reviews.append({
                    'user': review['userID'],
                    'item': review['hotelID'],
                    'text': review['reviewText'],
                    'rating': review['rating'],
                    'value': review['subRatings']['Value'],
                    'location': review['subRatings']['Location'],
                    'sleep_quality': review['subRatings']['Sleep Quality'],
                    'rooms': review['subRatings']['Rooms'],
                    'cleanliness': review['subRatings']['Cleanliness'],
                    'service': review['subRatings']['Service'],
                })
        return processed_reviews
    except FileNotFoundError:
        print(f"[Error] File not found: {filepath}")
    except json.JSONDecodeError as e:
        print(f"[Error] JSON Decode Error: {e}")
    except Exception as e:
        print(f"[Error] Unexpected error: {e}")
    return []


def save_csv(data: List[Dict], filepath: str):
    """Save list of dictionaries to a CSV file."""
    if not data:
        print(f"[Warning] No data to write to {filepath}")
        return
    with open(filepath, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"[Success] CSV saved: {filepath}")


def load_explanations(filepath: str) -> List[Dict]:
    """Load explanation data from a pickle file."""
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        explanations = []
        for line in data:
            explanations.append({
                'user': line['user'],
                'item': line['item'],
                'explanation': line['template'][2]
            })
        return explanations
    except FileNotFoundError:
        print(f"[Error] Pickle file not found: {filepath}")
    except Exception as e:
        print(f"[Error] Failed to load pickle file: {e}")
    return []


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, output_path: str):
    """Merge two dataframes on user and item, remove duplicates, and save."""
    merged = pd.merge(df1, df2, on=['user', 'item'], how='inner')
    merged = merged.drop_duplicates(subset=['user', 'item'], keep='first')
    merged.to_csv(output_path, index=False)
    print(f"[Success] Merged CSV saved: {output_path}")


# =============================== #
#              Main               #
# =============================== #

if __name__ == "__main__":
    # Step 1: Load and process review data
    reviews = load_and_process_reviews(REVIEW_JSON_PATH)
    save_csv(reviews, RAW_CSV_PATH)
    raw_df = pd.DataFrame(reviews)

    # Step 2: Load and save explanation data
    explanations = load_explanations(EXPLANATION_PICKLE_PATH)
    save_csv(explanations, EXPLANATION_CSV_PATH)
    explanation_df = pd.DataFrame(explanations)

    # Step 3: Merge both datasets and save
    merge_datasets(explanation_df, raw_df, MERGED_CSV_PATH)
