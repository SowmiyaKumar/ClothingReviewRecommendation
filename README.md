# NLP-Based Clothing Review Recommendation System

**Course:** Advanced Programming for Data Science (RMIT University)
**Type:** Individual Assignment (Assessment 3 – Milestone I & II)
**Author:** Sowmiya Kumar

---

## 📌 Project Overview

This project implements an **end-to-end NLP-driven recommendation system** that predicts whether a customer recommends a clothing item based on written reviews.

The work spans **two tightly integrated milestones**:

* **Milestone I – NLP & Machine Learning Pipeline**

  * Text preprocessing, feature engineering, and model comparison.
* **Milestone II – Flask Web Application**

  * Deployment of the trained NLP model into a user-facing recommendation website.

Together, the project demonstrates how **unstructured text → ML inference → deployed data product** can be built and evaluated end to end.

---

## 🎯 Problem Statement

Online clothing retailers receive thousands of unstructured customer reviews.
The objective of this project was to:

* Predict whether a review **recommends** a product
  (`0 = Not Recommended`, `1 = Recommended`)
* Evaluate whether **more textual information improves prediction accuracy**
* Compare multiple **feature representations**
* Deploy the best-performing model in a **real web application**

---

## 📂 Project Structure

```
.
├── app/
│   ├── app.py                         # Flask application entry point
│   ├── templates/                     # Jinja2 HTML templates
│   └── static/                        # CSS and static assets
│
├── assignment3_ll/
│   ├── task1.ipynb                    # Text preprocessing & vocabulary creation
│   ├── task2_3.ipynb                  # Feature engineering & ML experiments
│
├── assignment3.csv                    # Original dataset
├── processed.csv                      # Cleaned & processed dataset
├── assignment3_II.csv                   # App dataset (titles & descriptions)
│
├── fastText.model                     # Pre-trained FastText embeddings
├── fastText.model.wv.vectors_ngrams.npy
├── logistic_regression_model_tfidf_fasttext.pkl
├── tfvectorizer.pkl
├── count_vectors.txt
├── vocab.txt
│
├── Demo_video_s4040536/                # Application demo video
├── README.md
├── requirements.txt
```

---

## 🧠 Milestone I – NLP & Machine Learning Pipeline

### Dataset

The original dataset contains **customer clothing reviews** with the following fields:

* Clothing ID
* Age
* Title
* Review Text
* Rating
* Recommended IND (target)
* Positive Feedback Count
* Division Name
* Department Name
* Class Name

### Text Preprocessing

* Regex-based tokenisation

  ```
  [a-zA-Z]+(?:[-'][a-zA-Z]+)?
  ```
* Lowercasing and minimum token-length filtering
* Stopword removal using provided list
* Noise reduction:

  * Removed hapax legomena (single-occurrence tokens)
  * Removed top 20 most frequent terms by document frequency

### Vocabulary & Feature Files

* Clean unigram vocabulary built and saved
* Sparse count-vector representations persisted
* TF-IDF vectorizer serialized for reuse in deployment

### Feature Engineering Strategies

1. **Bag-of-Words (Count Vector)**
2. **Unweighted FastText Embeddings**
3. **TF-IDF Weighted FastText Embeddings**

Each document vector is constructed either by raw counts or by averaging word embeddings (with and without TF-IDF weighting).

---

## 📊 Model Experiments & Results

### Evaluation Setup

* Binary classification task:

  * `0` → Not Recommended
  * `1` → Recommended
* Model:

  * Logistic Regression
* Validation:

  * 5-fold cross-validation
* Metrics:

  * Accuracy, Precision, Recall, F1-score

### Does More Information Improve Accuracy?

| Input Text          | Feature Type    | Accuracy  | Precision | Recall | F1        |
| ------------------- | --------------- | --------- | --------- | ------ | --------- |
| Review Text         | Count Vector    | **0.875** | 0.903     | 0.949  | **0.926** |
| Review Text         | TF-IDF FastText | 0.843     | 0.861     | 0.963  | 0.909     |
| Title Only          | TF-IDF FastText | 0.830     | 0.854     | 0.947  | 0.898     |
| Title + Review Text | Count Vector    | **0.890** | **0.916** | 0.952  | **0.934** |
| Title + Review Text | TF-IDF FastText | 0.860     | 0.882     | 0.957  | 0.918     |

### Key Insights

* **Combining Review Title + Review Text consistently improves performance**
* TF-IDF weighted embeddings outperform unweighted embeddings
* Count-based models perform well but lack semantic understanding
* Embedding-based models generalize better to nuanced language

---

## 🌐 Milestone II – Flask Web Application

### Application Capabilities

The Flask application transforms the trained NLP model into a **user-facing product recommendation system**.

#### Core Features

* Browse clothing items
* Search items by keyword or category
* View item details and existing reviews
* Create a new review via web form
* Automatic ML-based recommendation prediction

#### ML Integration

* Loads serialized TF-IDF vectorizer and trained Logistic Regression model
* Applies the **exact same preprocessing pipeline** as Milestone I
* Predicts recommendation label in real time
* Displays prediction clearly to the user

#### Human-in-the-Loop Design

* Users can **override the model’s recommendation**
* Final label is saved and reflected in the application
* Demonstrates responsible ML usage

#### End-to-End Flow

```
User review →
Text preprocessing →
TF-IDF + FastText encoding →
Model inference →
Recommendation label →
UI display →
Persisted review
```

### Demo

A demo video is included showing:

* Browsing and searching clothing items
* Submitting a new review
* Viewing predicted recommendation
* Seeing the new review rendered on the site

---

## ▶️ How to Run the Application

```bash
pip install -r requirements.txt
python app/app.py
```

The application runs locally and can be accessed via browser.

---

## 🧰 Tech Stack

* **Languages:** Python
* **Libraries:**

  * pandas, numpy
  * scikit-learn
  * FastText, gensim
  * regex
* **Web:** Flask, Jinja2, HTML/CSS
* **Tools:** Jupyter Notebook

---

## 🚀 Skills Demonstrated

* NLP preprocessing and feature engineering
* Word embeddings & TF-IDF integration
* Model evaluation with cross-validation
* Serialization of ML pipelines
* Flask-based ML deployment
* End-to-end ML product development

---

## 📄 Notes

* Datasets and assignment specifications are provided by **RMIT University**
* Used strictly for academic and portfolio demonstration
* Focused on technical rigour and real-world applicability



