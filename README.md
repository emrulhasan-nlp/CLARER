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

---

Your directory should look like this:

.
├── rawdata/
│ ├── tripAdvisor/
│ ├── amazonTV/
│ └── yelp/
├── trip_preprocessing.py
├── amztv_preprocessing.py
├── yelp_preprocessing.py
├── NRTPlus.py
└── README.md

---

### Step 2: Preprocess the Datasets

Run the following scripts to preprocess the datasets. These scripts generate intermediate data required for model training.

```bash
python trip_preprocessing.py
python amztv_preprocessing.py
python yelp_preprocessing.py
```
