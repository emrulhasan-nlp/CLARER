# CLARER

**Contrastive Learning for Aspect Representation towards Explainable Recommendation**

CLARER is a model that utilizes contrastive learning to generate aspect-based representations from user reviews. These representations are then used for explainable recommendation tasks across different domains such as hotels, TVs, and restaurants.

---

## 🚀 How to Run the Model

### Step 1: Download the Dataset

Download the dataset from the following Google Drive link:  
🔗 [Dataset Link](https://drive.google.com/drive/folders/1yB-EFuApAOJ0RzTI0VfZ0pignytguU0_)

Place the `review.pickle` file inside each of the following folders:

- `tripAdvisor`
- `amazonTV`
- `yelp`

Your directory should look like this:

```
.
├── rawdata/
│   ├── tripAdvisor/
│   │   ├── OriginalReviews.json
│   │   └── review.pickle
│   ├── amazonTV/
│   │   ├── Movies_and_TV.json
│   │   └── review.pickle
│   └── yelp/
│       ├── yelp_academic_dataset_review.json
│       └── review.pickle
├── prepdata/
│   ├── tripAdvisor/
│   │   └── tripdata.csv
│   ├── amazonTV/
│   │   └── amz_tv.csv
│   └── yelp/
│       └── yelp.csv
├── trip_preprocessing.py
├── amztv_preprocessing.py
├── yelp_preprocessing.py
├── NRTPlus.py
└── README.md
```

---

### Step 2: Preprocess the Datasets

Run the following scripts to preprocess the datasets. These scripts generate intermediate data required for model training:

```bash
python trip_preprocessing.py
python amztv_preprocessing.py
python yelp_preprocessing.py
```

---

### Step 3: Run the Model

After preprocessing, use the following command to run the model:

```bash
python NRTPlus.py --dataset yelp.csv
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
├── *.py              # Scripts for preprocessing and training
├── README.md         # This file
└── requirements.txt  # Dependencies
```

---

## 📫 Contact

If you have any questions, suggestions, or want to contribute, feel free to open an issue or submit a pull request.

---

<!-- ## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details. -->
