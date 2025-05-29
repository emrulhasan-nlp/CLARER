import pickle 
import pandas as pd
import json
import csv

# Yelp review file that contains explanation but original review is missing
yelp_path="../rawdata/yelp/reviews.pickle"

with open(yelp_path, 'rb') as f:
    yelp_reviews = pickle.load(f)

yelp_revdf=pd.DataFrame(yelp_reviews)

yelp_revdf=yelp_revdf.drop(columns=['predicted'])
explanations=[exp for _, _, exp, _ in yelp_revdf['template']]
yelp_revdf['explanation']=explanations
yelp_revdf.drop(columns=['template'], inplace=True)

############Original Review #################
def convert_json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file, \
            open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['user_id', 'business_id', 'text', 'stars'])  # Header row

        for line in json_file:
            try:
                data = json.loads(line)
                user_id = data.get('user_id', '')
                business_id = data.get('business_id', '')
                text = data.get('text', '')
                stars = data.get('stars', '')
                csv_writer.writerow([user_id, business_id, text, stars])
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

# Convert the json to csv file
json_file_path = '../rawdata/yelp/yelp_academic_dataset_review.json'
csv_file_path = '../rawdata/yelp/yelp_review.csv'
convert_json_to_csv(json_file_path, csv_file_path)
print(f"Conversion complete. CSV file saved to: {csv_file_path}")

df=pd.read_csv('../rawdata/yelp/yelp_review.csv')
df=df.rename(columns={'user_id':'user', "business_id":'item', 'stars': 'rating'})

merged_df = pd.merge(yelp_revdf,df, on=['user', 'item'], how='inner')
merged_df.to_csv('../prepdata/yelp/yelp.csv', index=False)
