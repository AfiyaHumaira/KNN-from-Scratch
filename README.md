# KNN-from-Scratch
KNN classification of Iris and custom news dataset using Python from scratch

# 🧠 KNN from Scratch: Flower and News Classification with Custom Evaluation Metrics

This project demonstrates how to build a **K-Nearest Neighbors (KNN)** classifier from scratch using basic Python. The classifier is evaluated on two datasets:

- 🌸 **Iris Flower dataset** (a standard dataset in ML)
- 📰 **Custom News dataset** (sports vs politics headlines)

The project includes:
- Manual implementation of the KNN algorithm
- Manual implementation of evaluation metrics: accuracy, confusion matrix, precision, recall, F1-score
- Automatic selection of best `k` and best train-test split ratio
- Comparison with **Scikit-learn's KNN**

---

## 📁 Files Included

| File Name                                                   | Description                                                  |
|-------------------------------------------------------------|--------------------------------------------------------------|
| `KNN_from_Scratch_Iris_and_News_Classification.ipynb`       | Main Google Colab notebook with full code and outputs        |
| `news_dataset_200.csv`                                      | Custom dataset: 100 sports + 100 politics news headlines     |
| `README.md`                                                 | This documentation file                                      |

---

## 🚀 How to Use

1. Open the `.ipynb` notebook in [Google Colab](https://colab.research.google.com/)
2. Upload the `news_dataset_200.csv` file in the Colab session
3. Run all cells to:
   - Train and test the scratch KNN on both datasets
   - Evaluate with custom metrics
   - Compare with Scikit-learn KNN

---

## 📊 Dataset Overview

### 🌸 Iris Dataset
- Features: sepal length, sepal width, petal length, petal width
- Classes: setosa, versicolor, virginica
- Total samples: 150

### 📰 News Dataset
- 100 sports headlines
- 100 politics headlines
- Preprocessed into numerical form using `CountVectorizer`

---

## ⚙️ Custom KNN Algorithm

KNN is a lazy learning, instance-based algorithm that classifies a data point by finding the majority class among its *k* nearest neighbors using a distance metric (Euclidean distance).

### 🔢 Euclidean Distance Formula

```math
d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}

