# Credit-Card-Fraud-Detection
Machine Learning-Based Credit Card Fraud Detection Using Python and Random Forest

## Project Overview

This project explores how supervised machine learning can identify fraudulent transactions in highly imbalanced financial datasets.

The model was trained and evaluated on a public dataset containing 284,807 credit card transactions, including 492 fraudulent transactions.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab

## Methodology

1. Loaded and explored the transaction dataset.
2. Split the data into 60% training, 20% validation, and 20% testing.
3. Trained a Random Forest classifier with 100 decision trees and balanced class weights.
4. Compared classification thresholds using validation F1 scores.
5. Selected a threshold of 0.3.
6. Evaluated the final model on an untouched test set.

## Final Test Results

| Metric | Score |
|---|---|
| Accuracy | 99.96% |
| Precision | 94.19% |
| Recall | 81.82% |
| F1 Score | 87.57% |

### Confusion Matrix

![Final Confusion Matrix](final_confusion_matrix.png)

The model successfully detected 81 of 99 fraudulent transactions in the test set, with 5 false positives and 18 false negatives.

### Threshold Analysis

![Threshold Comparison](threshold_comparison.png)

An exploratory threshold comparison demonstrated the tradeoff between precision and recall. The final threshold was selected separately using validation data.

## Key Takeaways

- Accuracy alone can be misleading when dealing with imbalanced datasets.
- Precision and recall are important for evaluating fraud detection systems.
- Adjusting the classification threshold can improve fraud detection performance.
- Separating training, validation, and test data helps prevent overly optimistic model evaluation.

## Dataset

Credit Card Fraud Detection Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Notebook

[View the Python Notebook](Credit_Card_Fraud_Detection.ipynb)

## Limitations

The model was evaluated on a historical, anonymized dataset. Its performance does not guarantee effectiveness on live financial transactions.

Additional testing on newer datasets would be necessary before real-world deployment.
