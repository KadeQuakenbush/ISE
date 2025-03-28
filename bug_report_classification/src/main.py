########## 1. Import required libraries and debugging ##########

import requests
import json
import pandas as pd
import numpy as np
import re
import math
import time
import warnings

import os
import subprocess

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, precision_recall_curve, auc, silhouette_score

# Data resampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter

# Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Import preprocessing functions from utils
from utils import preprocess, vectorize_text, classify


# ========== Debugging ==========

debug_mode = True

def debug(message):
    if debug_mode:
        print("[DEBUG]", message)

warnings.filterwarnings("ignore") # Prevents warning when number of cores cannot be found



########### 2. Method configuration ##########
'''
Feature extraction options: TF-IDF (baseline), TF
Classifier options: Naive Bayes (baseline), Logistic Regression
'''
dataset_size = 100000

vec_options = {
    "tf": "TF",
    "tfidf": "TF-IDF",
    "tfigm": "TF-IGM"
}
clf_options = {
    "nb": "Multinomial Naive Bayes",
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "svm": "Support Vector Machine",
    "kmeans": "k-Means Clustering"
}
unsupervised_clfs = ["kmeans"]

vec_name = "tfidf" # Options: tf, tfidf (baseline), tfigm
clf_name = "nb" # Options: nb (baseline), lr, rf, svm, kmeans (not measured)
testing_clfs = ["nb", "lr"]



########## 3. Download and read the dataset ##########

base_path = "C:/Users/jedim/OneDrive/Documents/Work/Uni/Canvas Work/Intelligent_Software_Engineering/ISE_priv/ISE/bug_report_classification"

url = "https://bugzilla.mozilla.org/rest/bug" # Bugzilla API
params = {
    "product": "Firefox", # Specify product (e.g., Firefox, Thunderbird)
    "component": "General", # Component within the product
    "include_fields": "id,severity,summary,description,comments", # Fields to retrieve
    "is_confirmed": True,
    "limit": min(dataset_size, 10000),
    "offset": 0
}
bugs = []

while len(bugs) < dataset_size:
    try:
        response = requests.get(url, params=params)
        data = response.json()

        if "bugs" in data and data["bugs"]:
            bugs.extend(data["bugs"])
            params["limit"] = min(dataset_size - len(bugs), 10000)
            params["offset"] += 10000
            debug(f"Len bugs: {len(bugs)}")
        else:
            debug("No more bugs found.")
            break

    except requests.exceptions.RequestException as e:
        debug(f"Request failed: {e}")
        break

pd_all = pd.DataFrame(bugs)
pd_all = pd_all[["id", "severity", "summary", "description", "comments"]] # Sort
debug(f"DataFrame shape after fetching: {pd_all.shape}")
debug(f"Distribution: {pd_all["severity"].value_counts()}")

pd_all.to_csv(f"{base_path}/datasets/mozilla_bugs.csv", index=False)
debug(f"Fetched {len(pd_all)} bug reports and saved to mozilla_bugs.csv")

# pd_all = pd.read_csv(f"{base_path}/datasets/mozilla_bugs.csv")
pd_all = pd_all.sample(frac=1, random_state=999).head(dataset_size) # Shuffle and take only the first n records
pd.set_option("display.max_columns", None) # Show all columns
debug(f"DataFrame shape after reading CSV: {pd_all.shape}")
debug(f"DataFrame head after reading CSV: \n{pd_all.head()}")

# Merge summary and description into a single column; if description is NaN, use summary only

pd_all["comments"] = pd_all.apply(
    lambda row: len(row["comments"]) if isinstance(row["comments"], list) else 0,
    axis=1
)
debug(f"DataFrame shape after counting comments: {pd_all.shape}")
debug(f"DataFrame head after counting comments: \n{pd_all.head()}")

pd_all["summary+description"] = pd_all.apply(
    lambda row: f"{row["summary"]}. {row["description"]}" if pd.notna(row["description"]) else f"{row["summary"]}",
    axis=1
)
debug(f"DataFrame shape after merging summary and description: {pd_all.shape}")
debug(f"DataFrame head after merging summary and description: \n{pd_all.head()}")

# Rename column

pd_tplusb = pd_all.rename(columns={
    "summary+description": "text",
    "comments": "comment_count"
})
debug(f"DataFrame shape after renaming columns: {pd_tplusb.shape}")
debug(f"DataFrame head after renaming columns: \n{pd_tplusb.head()}")

pd_tplusb.to_csv(f"{base_path}/datasets/reshaped_mozilla_bugs.csv", index=False, columns=["id", "severity", "text", "comment_count"])



########## 4. Configure parameters ##########

# Get samples and set results file

datafile = f"{base_path}/datasets/reshaped_mozilla_bugs.csv" # Data file to read
out_csv_name = f"{base_path}/results/mozilla_bugs_results.csv" # Output CSV file

# Replace NaN's with empty string

data = pd.read_csv(datafile).fillna("")
debug(f"DataFrame shape after data cleaning: {data.shape}")
debug(f"DataFrame head after data cleaning: \n{data.head()}")

original_data = data.copy() # Keep a copy for referencing original data if needed

# Text preprocessing

data["text"] = data["text"].apply(preprocess)
debug(f"DataFrame shape after text cleaning: {data.shape}")
debug(f"DataFrame head after text cleaning: \n{data.head()}")

# Numerically label severities and remove any records that have invalid severities

severity_mapping = {
    "S1": 4, "blocker": 4, "critical": 4,
    "S2": 3, "major": 3,
    "S3": 2, "normal": 2,
    "S4": 1, "minor": 1, "trivial": 1,
    "N/A": 0, "enhancement": 0,
}
data["severity"] = data["severity"].map(severity_mapping)
data = data.dropna(subset=["severity"])
debug(f"DataFrame shape after severity mapping: {data.shape}")
debug(f"DataFrame head after severity mapping: \n{data.head()}")

# Save preprocessed data to CSV
data.to_csv(f"{base_path}/datasets/preprocessed_mozilla_bugs.csv", index=False, columns=["id", "severity", "text", "comment_count"])



########## 5. Train the model and evaluate ##########

for clf_name in ["nb", "lr"]:
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Scaling comment count to reduce bias
    if clf_name in ["nb", "svm", "kmeans"]:
        data["comment_count"] = MinMaxScaler().fit_transform(data[["comment_count"]])
    elif clf_name in ["lr"]:
        data["comment_count"] = StandardScaler().fit_transform(data[["comment_count"]])

    X = data[["text", "comment_count"]]
    y = data["severity"]

    # Lists to store metrics across folds
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    f1_weighted_scores = []

    start_time = time.time()

    # Stratified K-Fold Cross-Validation

    for fold, (train_index, test_index) in enumerate(skf.split(data["text"], data["severity"])):
        print(f"Fold {fold + 1}/{n_splits} running...")

        # Splitting and vectorizing the data
        processed_data = vectorize_text(X, y, vec_name, train_index, test_index)

        # Training and testing the classifier
        y_test, y_pred = classify(processed_data, clf_name, fold)
        debug(f"Predicted severity distribution: {Counter(y_pred)}")

        if clf_name not in unsupervised_clfs:
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            precisions.append(prec)

            rec = recall_score(y_test, y_pred, average="macro")
            recalls.append(rec)

            f1 = f1_score(y_test, y_pred, average="macro")
            f1_scores.append(f1)

            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            f1_weighted_scores.append(f1_weighted)

    end_time = time.time()
    elapsed_time = end_time - start_time
    debug(f"Elapsed time: {elapsed_time:.2f} seconds")

    # --- Aggregate results ---
    final_accuracy = np.mean(accuracies)
    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    final_f1 = np.mean(f1_scores)
    final_f1_weighted = np.mean(f1_weighted_scores)

    print(f"=== {clf_options[clf_name]} + {vec_options[vec_name]} Results ===")
    print(f"Number of folds:     {n_splits}")
    print(f"Average Accuracy:    {final_accuracy:.4f}")
    print(f"Average Precision:   {final_precision:.4f}")
    print(f"Average Recall:      {final_recall:.4f}")
    print(f"Average F1:          {final_f1:.4f}")
    print(f"Average F1 Weighted: {final_f1_weighted:.4f}")

    # Save final results to CSV (append mode)

    try:
        existing_data = pd.read_csv(out_csv_name, nrows=1)
        header_needed = False
    except:
        header_needed = True

    df_log = pd.DataFrame(
        {
            "method": [clf_options[clf_name] + " + " + vec_options[vec_name]],
            "folds": [n_splits],
            "Accuracy": [final_accuracy],
            "Precision": [final_precision],
            "Recall": [final_recall],
            "F1 Score": [final_f1],
            "F1 Weighted Score": [final_f1_weighted]
        }
    )

    df_log.to_csv(out_csv_name, mode="a", header=header_needed, index=False)

    print(f"\nResults have been saved to: {out_csv_name}")