# Wine Classification Machine Learning Pipeline

An end-to-end Machine Learning pipeline built using Python and Scikit-learn to classify wines into three distinct cultivars based on chemical analysis. 

The pipeline benchmarks six classification algorithms using 10-fold cross-validation, automatically selects the best performer, evaluates it on a test set, and performs real-time inference on new samples.

---

## Project Overview

- **Dataset**: UCI Wine Dataset (`sklearn.datasets.load_wine`)
- **Total Samples**: 178 bottles
- **Features**: 13 continuous chemical properties (Alcohol, Malic acid, Flavanoids, Proline, Color intensity, etc.)
- **Target**: 3 Grape Cultivars (`Class 0`, `Class 1`, `Class 2`)

---

## Pipeline Workflow

1. **Data Ingestion & Preparation**: Load dataset into Pandas DataFrame and separate feature matrix ($X$) and target vector ($Y$).
2. **Train/Test Split**: 80% training data, 20% validation split with a fixed random seed.
3. **Cross-Validation Benchmarking**: 10-Fold Cross-Validation evaluated across six algorithms:
   - Logistic Regression (`LR`)
   - Linear Discriminant Analysis (`LDA`)
   - K-Nearest Neighbors (`KNN`)
   - Decision Tree (`CART`)
   - Gaussian Naive Bayes (`NB`)
   - Support Vector Classifier (`SVM`)
4. **Automated Model Selection**: Dynamically identifies the algorithm with the highest cross-validation mean accuracy.
5. **Evaluation**: Final training on the training split, followed by evaluation on the test split using Confusion Matrix and Classification Report (Precision, Recall, F1-Score).
6. **Inference**: Prediction verification against new custom chemical sample profiles.

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/wine-classification-pipeline.git](https://github.com/your-username/wine-classification-pipeline.git)
cd wine-classification-pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Script
```bash
python main.py
```

---

## Evaluation Metrics

- **Confusion Matrix**: Maps predicted classes vs. actual ground truth to highlight misclassifications.
- **Classification Report**: Details precision, recall, and F1-score for each cultivar class.
- **Inference Verification**: Tested with manual edge cases to validate class boundaries.