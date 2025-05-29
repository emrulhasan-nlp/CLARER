import json
import string
import pandas as pd
import csv
##############Get the review data###############
def is_english_simple(text):
    # Define English letters and common punctuation
    english_chars = set(string.ascii_letters + string.whitespace + string.punctuation)
    # Check if all characters in the text are part of English characters
    return all(char in english_chars for char in text)
def align_subratings(reviews, predefined_subratings):
    """
    Aligns subratings in a list of review dictionaries by filling missing subratings with 0.

    :param reviews: List of dictionaries, each containing a 'subRatings' dictionary
    :param predefined_subratings: List of predefined subrating categories
    :return: List of dictionaries with aligned subratings
    """

    subratings = reviews.get('subRatings', {})
    # Fill missing subratings with 0
    aligned_subratings = {key: subratings.get(key, 0) for key in predefined_subratings}
    reviews['subRatings'] = aligned_subratings

    return reviews

# Example usage
predefined_subratings = ['Value', 'Location', 'Sleep Quality', 'Rooms', 'Cleanliness', 'Service']

reviews ={'subRatings': {'Location': 5, 'Rooms': 5, 'Service': 5}}
aligned_reviews = align_subratings(reviews, predefined_subratings)
review_dir='../rawdata/tripAdvisor/OriginalReviews.json'
try:
    with open(review_dir, 'r') as f:
        try:
            reviews=[]
            data = json.load(f)

            for i, line in enumerate(data):
              #print(line)
              line=align_subratings(line, predefined_subratings)

              text=line['reviewText']
              user=line['userID']
              item=line['hotelID']
              review_text=line['reviewText']
              rating=line['rating']
              value=line['subRatings']['Value']
              location=line['subRatings']['Location']
              sleep_quality=line['subRatings']['Sleep Quality']
              rooms=line['subRatings']['Rooms']
              cleanliness=line['subRatings']['Cleanliness']
              new_dct={'user':user,'item':item,'text':review_text, 'rating':rating, 'value':value, 'location':location, 'sleep_quality':sleep_quality, 'rooms':rooms, 'cleanliness':cleanliness}
              reviews.append(new_dct)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
except FileNotFoundError:
    print(f"Error: File not found at {review_dir}")
except Exception as e:
    print(f"An error occurred: {e}")

# Specify the file name
import csv
file_name = "../rawdata/tripAdvisor/rawreview.csv"

# Writing to the CSV file
with open(file_name, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=reviews[0].keys())
    writer.writeheader()  # Write the header (field names)
    writer.writerows(reviews)  # Write the data rows

print(f"CSV file '{file_name}' has been created successfully!")
raw_df=pd.read_csv('rawreview.csv')


##########Explanation data############

filPath="../rawdata/tripAdvisor/reviews.pickle"

with open(filPath, 'rb') as file:
    # Load the data using pickle
    import pickle
    data = pickle.load(file)
explanations=[]
for line in data:
  user=line['user']
  item=line['item']

  xpln = line['template'][2]
  new_dct={'user':user,'item':item,'explanation':xpln}
  explanations.append(new_dct)

file_name = "../rawdata/tripAdvisor/tripdata.csv"

# Writing to the CSV file
with open(file_name, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=explanations[0].keys())
    writer.writeheader()  # Write the header (field names)
    writer.writerows(explanations)  # Write the data rows

print(f"CSV file '{file_name}' has been created successfully!")
df=pd.read_csv('../rawdata/tripAdvisor/tripdata.csv')

################## Merge explanation and review data to get the clean trip advisor data with user, item, review, rating, criteria rating, and explanation############

# Merge df and raw_df based on common columns ('user' and 'item')
merged_df = pd.merge(df, raw_df, on=['user', 'item'], how='inner')
merged_df = merged_df.drop_duplicates(subset=['user', 'item'], keep='first')

merged_file_name = "../prepdata/trip_review_xplns.csv"

# Save the merged DataFrame to a CSV file
merged_df.to_csv(merged_file_name, index=False)  # index=False prevents writing row indices

print(f"Merged DataFrame saved to '{merged_file_name}' successfully!")
