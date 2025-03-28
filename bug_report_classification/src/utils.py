import numpy as np

# Preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Text and feature engineering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# Visualising
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")

debug_mode = False

def debug(str):
    if debug_mode:
        print("[DEBUG]", str)

# Stopwords
NLTK_stop_words_list = stopwords.words("english")
custom_stop_words_list = ["..."]  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

# Define text preprocessing methods
def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r"", text)

def remove_build_identifier(text):
    """Remove lines that begin with 'Build Identifier:'."""
    return re.sub(r"(?m)^Build Identifier:.*\n?", "", text)

def remove_reproducible_section(text):
    """Remove everything from 'Reproducible:' onwards."""
    return re.sub(r"Reproducible:.*", "", text).strip()

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

# Lemmatiser
lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    # Tokenise
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in NLTK_stop_words_list]
    # Lemmatise
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = " ".join(tokens)
    return text

def preprocess(text):
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_build_identifier(text)
    text = remove_reproducible_section(text)
    text = remove_stopwords(text)
    text = clean(text)
    text = lemmatize(text)
    return text

def compute_igm(X, y):
    num_classes = len(np.unique(y))
    vectorizer = CountVectorizer(ngram_range=(1,2), max_features=1000)
    term_doc_matrix = vectorizer.fit_transform(X)
    terms = vectorizer.get_feature_names_out()

    igm_scores = np.zeros(len(terms))

    for term_idx, term in enumerate(terms):
        term_column = term_doc_matrix[:, term_idx].toarray().flatten()
        gravity = 0
        for class_label in range(num_classes):
            class_indices = np.where(y == class_label)[0]
            class_term_freq = term_column[class_indices].sum()
            class_total_freq = term_column.sum()
            if class_total_freq > 0:
                gravity += (class_term_freq / class_total_freq) * (class_label + 1)
        igm_scores[term_idx] = gravity

    igm_dict = dict(zip(terms, igm_scores))
    return igm_dict

def vectorize_text(X, y, vec_name, train_index, test_index):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    if vec_name == "tf":
        vectorizer = CountVectorizer(ngram_range=(1,2), max_features=1000)

        X = hstack([
            vectorizer.fit_transform(X["text"]),
            csr_matrix(X["comment_count"].values.reshape(-1, 1))
        ])
        X_train = hstack([
            vectorizer.fit_transform(X_train["text"]),
            csr_matrix(X_train["comment_count"].values.reshape(-1, 1))
        ])
        X_test = hstack([
            vectorizer.transform(X_test["text"]),
            csr_matrix(X_test["comment_count"].values.reshape(-1, 1))
        ])
    elif vec_name == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)

        X = hstack([
            vectorizer.fit_transform(X["text"]),
            csr_matrix(X["comment_count"].values.reshape(-1, 1))
        ])
        X_train = hstack([
            vectorizer.fit_transform(X_train["text"]),
            csr_matrix(X_train["comment_count"].values.reshape(-1, 1))
        ])
        X_test = hstack([
            vectorizer.transform(X_test["text"]),
            csr_matrix(X_test["comment_count"].values.reshape(-1, 1))
        ])
    elif vec_name == "tfigm":
        vectorizer = CountVectorizer(ngram_range=(1,2), max_features=1000)
        igm_scores = compute_igm(X["text"], y)
        sorted_terms = sorted(igm_scores, key=igm_scores.get, reverse=True)
        top_terms = sorted_terms[:1000]  # Select top 1000 terms based on IGM scores
        vectorizer = CountVectorizer(vocabulary=top_terms)

        X = hstack([
            vectorizer.fit_transform(X["text"]),
            csr_matrix(X["comment_count"].values.reshape(-1, 1))
        ])
        X_train = hstack([
            vectorizer.fit_transform(X_train["text"]),
            csr_matrix(X_train["comment_count"].values.reshape(-1, 1))
        ])
        X_test = hstack([
            vectorizer.transform(X_test["text"]),
            csr_matrix(X_test["comment_count"].values.reshape(-1, 1))
        ])
    else:
        raise ValueError("Invalid vectoriser selected.")

    return (X, X_train, X_test, y_train, y_test)

def classify(data_tuple, clf_name, fold):
    if clf_name in ["kmeans"]:
        return unsupervised_classify(data_tuple, clf_name, fold)
    return supervised_classify(data_tuple, clf_name, fold)

def supervised_classify(data_tuple, clf_name, fold):
    X, X_train, X_test, y_train, y_test = data_tuple

    if clf_name == "nb":
        clf = MultinomialNB()
        params = {
            "alpha": np.logspace(-3, 1, 5)
        }
    elif clf_name == "lr":
        clf = LogisticRegression(max_iter=1000)
        class_weights = {
            0: 1,
            1: 1.5,
            2: 2.5,
            3: 3.5,
            4: 4
        }
        params = {
            "C": np.logspace(-4, 4, 10),
            "penalty": ["l1", "l2"],
            "class_weight": ["balanced", class_weights]
        }
    elif clf_name == "rf":
        clf = RandomForestClassifier()
        params = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    elif clf_name == "svm":
        clf = SVC()
        params = {
            "C": np.logspace(-3, 3, 10),
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    else:
        raise ValueError("Unsupported classifier, choose a valid classifier.")

    grid = GridSearchCV(
        clf,
        params,
        cv=3,
        scoring="f1_macro"
    )
    grid.fit(X_train, y_train)
    debug(f"Best parameters (fold {fold}) {grid.best_params_}")
    best_clf = grid.best_estimator_
    best_clf.fit(X_train, y_train)

    y_pred = best_clf.predict(X_test)

    return y_test, y_pred

def unsupervised_classify(data_tuple, clf_name, fold):
    X, X_train, X_test, y_train, y_test = data_tuple

    if clf_name == "kmeans":
        n_clusters = 5
        clf = KMeans(n_clusters=n_clusters, random_state=fold)
        params = {
            "init": ["k-means++", "random"],
            "n_init": [10, 20, 30],
            "max_iter": [300, 500, 1000]
        }
    else:
        raise ValueError("Unsupported classifier, choose a valid classifier.")
    
    grid = GridSearchCV(
        clf,
        params,
        cv=3,
        scoring="f1_macro"
    )
    grid.fit(X)
    debug(f"Best parameters (fold {fold}) {grid.best_params_}")
    best_clf = grid.best_estimator_
    best_clf.fit(X_train, y_train)

    y_pred = best_clf.predict(X_test)

    # Visualize the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Fold {fold})")
    plt.show()

    return X, y_pred