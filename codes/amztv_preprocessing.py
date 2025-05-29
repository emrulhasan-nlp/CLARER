import pickle
import pandas as pd
import json
import csv
amazon_path="../rawdata/amazonMT/reviews.pickle" #######review data that has explanation field
filePath='../rawdata/amazonMT/Movies_and_TV.json' #Json file
csv_file_path = '../rawdata/amazonMT/Movies_and_TV.csv' # File that saves the json to csv
final_amzpath='../prepdata/AmazonMT/amz_mt.csv' #final cleaned and merged csv file

with open(amazon_path, 'rb') as f:
    amz_reviews = pickle.load(f)

amz_revdf=pd.DataFrame(amz_reviews)
amz_revdf.to_csv('../rawdata/amazonMT/amz_revdf.csv')

amz_revdf=amz_revdf.drop(columns=['Unnamed: 0'])
amz_revdf=amz_revdf.rename(columns={'asin': 'item'})

# Open the JSON file and the CSV file
with open(filePath, 'r', encoding='utf-8') as json_file, \
     open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:

    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['reviewerID', 'asin', 'reviewText', 'overall'])

    # Iterate over each line in the JSON file
    for line in json_file:
        try:
            # Parse each line as a JSON object
            data = json.loads(line)

            # Extract the required fields
            reviewerID = data.get('reviewerID', '')
            asin = data.get('asin', '')
            reviewText = data.get('reviewText', '')
            overall = data.get('overall', '')

            # Write the data to the CSV file
            csv_writer.writerow([reviewerID, asin, reviewText, overall])

        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")
            continue
print(f"Data written to {csv_file_path}")

raw_df=pd.read_csv('../rawdata/amazonMT/Movies_and_TV.csv')
raw_df=raw_df.rename(columns={'asin': 'item'})
raw_df=raw_df.rename(columns={'reviewerID': 'user'})
raw_df.drop_duplicates(inplace=True)
raw_df.rename(columns={'reviewText': 'text'}, inplace=True)

def merge_datasets(df1, df2):
    """
    Merge two datasets with different columns and sizes.

    Parameters:
    df1 (pandas.DataFrame): First dataset with columns: user, item, explanation, ratings
    df2 (pandas.DataFrame): Second dataset with columns: user, item, text, rating

    Returns:
    pandas.DataFrame: Merged dataset with columns: user, item, text, explanation, rating
    """
    # Merge the datasets on 'user' and 'item' columns
    merged_df = pd.merge(df1, df2, on=['user', 'item'], how='left')

    # Select and order the desired columns
    final_df = merged_df[['user', 'item', 'text', 'explanation', 'rating']]

    return final_df
final_df=merge_datasets(amz_revdf, raw_df)
merged_df = final_df.drop_duplicates(subset=['user', 'item'])
merged_df.reset_index(drop=True, inplace=True)

df_merged = amz_revdf.merge(raw_df[['user', 'item', 'text']], on=['user', 'item'], how='left')

merged_df.drop(columns=['predicted'], inplace=True)
merged_df.drop(columns=['overall'], inplace=True)
merged_df.rename(columns={'reviewText': 'text'}, inplace=True)
explanation=[]
for item in merged_df['template']:
  try:
    # Attempt to unpack assuming item is a tuple with 4 elements
    _, _, exp, _ = item
  except ValueError:
    # If unpacking fails due to too many values, try to evaluate as tuple
    try:
      item_tuple = eval(item)  # Evaluate the string as a Python expression (tuple)
      _, _, exp, _ = item_tuple
    except (ValueError, SyntaxError, TypeError):
      # Handle cases where evaluation fails or unexpected format
      print(f"Skipping invalid item: {item}")
      continue  # Move to the next item in the loop
  explanation.append(exp)
  
merged_df['explanation']=explanation
merged_df.drop(columns=['template'], inplace=True)
merged_df.to_csv(final_amzpath, index=False)
  

