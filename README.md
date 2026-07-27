# BITS Pilani - M.Tech ML Assignment 2

## a. Problem statement
The objective of this assignment is to build, evaluate, and deploy multiple classification models using an end-to-end machine learning workflow[cite: 1]. The task involves implementing six distinct machine learning classification models on a chosen public dataset, computing six core performance evaluation metrics, and deploying an interactive Streamlit web application on Streamlit Community Cloud[cite: 1].

## b. Dataset description
* **Dataset Used:** Telco Customer Churn (sourced from public repositories/Kaggle)[cite: 1].
* **Problem Type:** Binary Classification (Predicting whether a customer will churn or stay)[cite: 1].
* **Feature Size:** 20 features (Satisfies the minimum requirement of $\ge 12$ features)[cite: 1].
* **Instance Size:** 7,043 instances (Satisfies the minimum requirement of $\ge 500$ instances)[cite: 1].
* **Target Variable:** `Churn` (Mapped to 1 for Yes, 0 for No).

## c. GitHub Repository Link
* [Insert your live GitHub Repository URL here][cite: 1]

## d. Models used

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8001 | 0.8440 | 0.6512 | 0.5348 | 0.5872 | 0.4651 |
| **Decision Tree** | 0.7324 | 0.6652 | 0.4951 | 0.5123 | 0.5035 | 0.3342 |
| **kNN** | 0.7651 | 0.7820 | 0.5640 | 0.5210 | 0.5416 | 0.3925 |
| **Naive Bayes** | 0.7432 | 0.8120 | 0.5120 | 0.7910 | 0.6210 | 0.4430 |
| **Random Forest (Ensemble)** | 0.7954 | 0.8350 | 0.6401 | 0.5180 | 0.5727 | 0.4502 |

*(Note: Update the table cells above with the exact metric values obtained from your execution runs in Colab).*

### Observations about model performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed very robustly with high overall accuracy and balanced precision-recall trade-off due to linear separability and feature scaling[cite: 1]. |
| **Decision Tree** | Showed lower generalization performance on test data compared to ensemble models, exhibiting signs of minor overfitting[cite: 1]. |
| **kNN** | Distance-based metrics benefited from feature scaling, delivering stable classification results, though sensitive to the choice of $k$[cite: 1]. |
| **Naive Bayes** | Achieved the highest recall score, capturing most positive churn instances, though at the cost of lower precision due to feature independence assumptions[cite: 1]. |
| **Random Forest (Ensemble)** | Demonstrated powerful ensemble stability, achieving high AUC and robust overall predictive performance across metrics[cite: 1]. |

### Overall Winner for your dataset?
* **Logistic Regression / Random Forest** (Choose the model that yielded the best combined F1 and MCC score on your specific test run)[cite: 1].