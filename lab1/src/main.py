########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re
import math

import os
import subprocess

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc)

# Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Import preprocessing functions from utils
from utils import remove_html, remove_emoji, remove_stopwords, stem


########## DEBUGGING ########################

debug_mode = False

def debug(name, val):
    if debug_mode:
        print("--> DEBUG: ", name, val)


######### CONFIGURATION #####################
"""
Feature extraction options: TF-IDF (def), TF
Classifier options: Naive Bayes (def), Logistic Regression
"""


########## 2. Download & read data ##########

base_path = "C:/Users/jedim/OneDrive/Documents/Work/Uni/Canvas Work/Intelligent_Software_Engineering/ISE_priv/ISE/lab1"

# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = "tensorflow"
path = f"{base_path}/datasets/{project}.csv"
debug("Path", path)

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle
debug("DataFrame shape after reading CSV", pd_all.shape)
debug("DataFrame head after reading CSV", pd_all.head())

# Merge Title and Body into a single column; if Body is NaN, use Title only

pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)
debug("DataFrame shape after merging Title and Body", pd_all.shape)
debug("DataFrame head after merging Title and Body", pd_all.head())

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)

pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
debug("DataFrame shape after renaming columns", pd_tplusb.shape)
debug("DataFrame head after renaming columns", pd_tplusb.head())

pd_tplusb.to_csv(f"{base_path}/Title+Body.csv", index=False, columns=["id", "Number", "sentiment", "text"])


########## 4. Configure parameters & Start training ##########

# ========== Key Configurations ==========

datafile = f"{base_path}/Title+Body.csv" # Data file to read
REPEAT = 20 # Number of repeated experiments, more means more accuracte metrics (not better model)
out_csv_name = f"{base_path}/results/{project}_NB.csv" # Output CSV file

# ========== Read and clean data ==========

data = pd.read_csv(datafile).fillna('')
text_col = 'text'
debug("DataFrame shape after reading cleaned data", data.shape)
debug("DataFrame head after reading cleaned data", data.head())

original_data = data.copy() # Keep a copy for referencing original data if needed

# Text cleaning

data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords) # !!
data[text_col] = data[text_col].apply(stem)
debug("DataFrame shape after text cleaning", data.shape)
debug("DataFrame head after text cleaning", data.head())

# ========== Hyperparameter grid ==========

# Using logspace for var_smoothing: [1e-12, 1e-11, ..., 1]

params = { # !! For Naive Bayes
    'var_smoothing': np.logspace(-12, 0, 13)
}
# params = { # !! For Logistic Regression
#     'C': np.logspace(-4, 4, 20)  # Hyperparameter grid for Logistic Regression
# }

# Lists to store metrics across repeated runs

accuracies  = []
precisions  = []
recalls     = []
f1_scores   = []
auc_values  = []

for repetition in range(REPEAT):

    # --- 4.1 Split into train/test ---

    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.3, random_state=repetition # !! Test size affects training effectiveness
    )

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]
    debug(f"Train/Test split shapes (repeat {repetition})", (train_text.shape, test_text.shape))

    y_train = data['sentiment'].iloc[train_index].values
    y_test  = data['sentiment'].iloc[test_index].values
    debug(f"y_train shape and type (repeat {repetition})", (y_train.shape, type(y_train)))
    debug(f"y_test shape and type (repeat {repetition})", (y_test.shape, type(y_test)))

    # --- 4.2 TF vectorization ---

    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),  # !! Adjust n-gram range, wider range tends to be more accurate
        max_features=1000    # !! Adjust max features
    )
    tf = CountVectorizer(
        ngram_range=(1, 3),  # !! Adjust n-gram range, wider range tends to be more accurate
        max_features=1000    # !! Adjust max features
    )
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)
    
    # Convert sparse matrices to dense format

    X_train = X_train.toarray()
    X_test = X_test.toarray()
    debug(f"X_train shape and type (repeat {repetition})", (X_train.shape, type(X_train)))
    debug(f"X_test shape and type (repeat {repetition})", (X_test.shape, type(X_test)))
   
    # --- 4.3 Logistic Regression model & GridSearch ---

    clf = GaussianNB() # !!
    # clf = LogisticRegression(max_iter=1000) # !!
    grid = GridSearchCV(
        clf,
        params,
        cv=5,              # !! More folds is typically more effective, but more computationally expensive
        scoring='roc_auc'  # Using roc_auc as the metric for selection
    )
    grid.fit(X_train, y_train)
    debug(f"Best parameters (repeat {repetition})", grid.best_params_)

    # Retrieve the best model

    best_clf = grid.best_estimator_
    best_clf.fit(X_train, y_train)

    # --- 4.4 Make predictions & evaluate ---

    y_pred = best_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precisions.append(prec)

    rec = recall_score(y_test, y_pred, average='macro')
    recalls.append(rec)

    f1 = f1_score(y_test, y_pred, average='macro')
    f1_scores.append(f1)

    # AUC
    # If labels are 0/1 only, this works directly.
    # If labels are something else, adjust pos_label accordingly.
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    auc_val = auc(fpr, tpr)
    auc_values.append(auc_val)

# --- 4.5 Aggregate results ---

final_accuracy  = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall    = np.mean(recalls)
final_f1        = np.mean(f1_scores)
final_auc       = np.mean(auc_values)

print("=== Logistic Regression + TF Results ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

# Save final results to CSV (append mode)

try:
    # Attempt to check if the file already has a header
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame(
    {
        'repetitions': [REPEAT],
        'Accuracy': [final_accuracy],
        'Precision': [final_precision],
        'Recall': [final_recall],
        'F1': [final_f1],
        'AUC': [final_auc],
        'CV_list(AUC)': [str(auc_values)]
    }
)

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have been saved to: {out_csv_name}")