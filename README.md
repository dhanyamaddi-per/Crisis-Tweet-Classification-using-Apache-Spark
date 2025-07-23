# Crisis-Tweet-Classification-using-Apache-Spark

# 🧠 Social Media Text Analysis using Apache Spark

This project focuses on processing, analyzing, and visualizing social media text data using Apache Spark. The goal is to understand underlying sentiment trends, discover latent topics using unsupervised learning, and classify user responses with various machine learning models.

## 📁 Project Structure

- `AIT-614 final project.ipynb` – Jupyter Notebook version of the project.
- `AIT-614 final project.py` – Python script version of the notebook.
- `finalproject_wordCloudVisuals.ipynb` – Additional visualizations such as word clouds and frequency plots.
- `CLEAN_train.csv` / `CLEAN_test.csv` – Input datasets used for training and testing (stored on Databricks).
  
## 🎯 Objectives

- Clean and preprocess raw text data.
- Perform **topic modeling** using Latent Dirichlet Allocation (LDA).
- Build machine learning pipelines for classification using:
  - Logistic Regression
  - Random Forest
  - Naive Bayes
- Visualize model performance using confusion matrices and clustering results.
- Conduct **sentiment analysis** using VADER to classify sentiments into Positive, Negative, and Neutral.

## 🛠️ Technologies & Libraries

- **Apache Spark** (PySpark)
- **Databricks** for cloud-based Spark execution
- **VADER Sentiment Analysis**
- **scikit-learn**, **matplotlib**, **seaborn** for visualization and evaluation
- **NLP Modules**: Tokenizer, StopWordsRemover, TF-IDF, LDA

## 🔍 Key Features

### ✅ Preprocessing
- Tokenization and stop word removal using `pyspark.ml.feature`.
- Handling missing data and typecasting.

### 📊 Topic Modeling
- Extracted latent topics from text using LDA on Spark.
- Transformed datasets into topic distributions.

### 🤖 Machine Learning
- Built and compared multiple classifiers.
- Evaluated using Accuracy, Precision, Recall, and F1-score.
- Confusion matrices for model comparison.

### 💬 Sentiment Analysis
- Applied VADER to analyze emotional tone of tweets.
- Created bar chart and heatmap for sentiment distribution.

### 🔍 Clustering & PCA
- Used KMeans for clustering and PCA for dimensionality reduction.
- Visualized clusters in 2D using scatter plots.

## 📈 Output Visuals

- Confusion matrices (heatmaps)
- Topic distributions
- Sentiment bar charts
- PCA-based cluster plots
- Word cloud visualizations

## 🚀 How to Run

1. Run the notebook/script in a Databricks environment with Spark configured.
2. Ensure the CSV datasets are uploaded to Databricks `/FileStore`.
3. Install missing libraries (e.g., `vaderSentiment`) using `%pip install`.
4. Execute all cells to complete topic modeling, classification, and sentiment analysis.

## 📦 Example Sentiment Output

| Text Sample | Sentiment |
|-------------|-----------|
| "This is amazing!" | Positive |
| "I'm so frustrated..." | Negative |
| "It's okay." | Neutral |

---

