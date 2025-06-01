# CLARER

**Contrastive Learning for Aspect Representation towards Explainable Recommendation**

CLARER is a model that utilizes contrastive learning to generate aspect-based representations from user reviews. These representations are then used for explainable recommendation tasks across different domains such as hotels, TVs, and restaurants.

---

## 🚀 How to Run the Model

### Step 1: Download the Dataset

Download the dataset from the following Google Drive link:  
🔗 [Dataset Link](https://drive.google.com/drive/folders/1yB-EFuApAOJ0RzTI0VfZ0pignytguU0_)

Place the `review.pickle` file inside each of the following folders:

- `trip`
- `amzMT`
- `yelp`

Please note that the only tripadvisor dataset has both the explanation and original review data. However, for amazMT and yelp dataset has only explanation. Corresponding original review datasets can be found in https://business.yelp.com/data/resources/open-dataset/ and https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ for yelp and amazon review respectively.
Your directory should look like this:

```
.
├── rawdata/
│   ├── trip/
│   │   ├── OriginalReviews.json
│   │   └── review.pickle
│   ├── amzMT/
│   │   ├── Movies_and_TV.json
│   │   └── review.pickle
│   └── yelp/
│       ├── yelp_academic_dataset_review.json
│       └── review.pickle
├── prepdata/
│   ├── trip/
│   │   ├── trip.csv
|   |   └── trip_emb.npy
│   ├── amzMT/
│   │   ├── amzMT.csv
|   |   └── amz_emb.npy
│   └── yelp/
│       ├── yelp.csv
|       └── yelp_emb.npy
├──codes/
|    ├── trip_preprocessing.py
|    ├── amzMT_preprocessing.py
|    ├── yelp_preprocessing.py
|    ├── aspect_embedder.py
|    ├── CLARER_ratingPred.py
|    └── CLARER_exp.py
|--results/
│   ├── trip/
│   │   └── trip_model.pth
|   |   └── trip.logs
│   ├── amzMT/
│   │   └── amzMT_model.pth
|   |   └── amzMT.logs
│   └── yelp/
│       |── yelp_model.pth
|       └── yelp.logs
|
└── README.md
```

---

### Step 2: Preprocess the Datasets

Run the following scripts to preprocess the datasets. These scripts generate intermediate data required for model training:

```bash
python trip_preprocessing.py
python amzMT_preprocessing.py
python yelp_preprocessing.py
python aspect_embedder.py

```

---

### Step 3: Run the Model

After preprocessing, use the following command to run the model:

```bash
python CLARER_ratingPred.py --dataset yelp.csv
python CLARER_exp.py --dataset yelp.csv
```

> Replace `yelp.csv` with `tripdata.csv` or `amz_tv.csv` based on the dataset you're using.

---

## 📦 Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧠 Project Structure

```
CLARER/
├── rawdata/
├── prepdata/
├── codes/
├── results/
├── README.md
└── requirements.txt  # Dependencies
```

---

## 📫 Contact

If you have any questions, suggestions, or want to contribute, feel free to open an issue or submit a pull request.

---

<!-- ## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details. -->
