# Student Performance Prediction

## Summary

The goal of this project is to predict the final grade (G3) of students based on academic, demographic, and behavioral attributes from the UCI Student Performance dataset.

Two datasets were provided: one for Mathematics and one for Portuguese language performance. These datasets were merged based on 13 identifying columns to detect overlapping students. For overlapping entries, grades were averaged. Unique students from both datasets were also retained.

---

## Data Preprocessing

The following preprocessing steps were performed:

* **Binary Encoding:** Binary categorical values (e.g., yes/no, M/F, U/R) were mapped to 0/1.
* **One-Hot Encoding:** Nominal categorical features (e.g., `school`, `Mjob`, `Fjob`, `reason`, `guardian`) were converted using one-hot encoding.
* **Standardization:** Numeric features were standardized to have zero mean and unit variance.
* **Train-Test Split:** The dataset was split into 80% training and 20% testing subsets.

---

## Model Building

Three regression models were developed and evaluated:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**

Baseline performance was assessed using **5-fold cross-validation** on the training set. Hyperparameters for the Decision Tree and Random Forest models were tuned using **GridSearchCV**.

---

## Model Evaluation

Performance on the held-out test set:

| Model             | R²     | MAE    | RMSE   |
| ----------------- | ------ | ------ | ------ |
| Linear Regression | 0.9151 | 0.7133 | 0.9713 |
| Decision Tree     | 0.9150 | 0.7526 | 0.9717 |
| Random Forest     | 0.9217 | 0.6687 | 0.9328 |

---

## Feature Importance (Random Forest)

Top 10 most important features influencing the final grade (G3):

| Feature  | Importance |
| -------- | ---------- |
| G2       | 0.8202     |
| absences | 0.0340     |
| G1       | 0.0187     |
| freetime | 0.0084     |
| goout    | 0.0084     |
| failures | 0.0078     |
| age      | 0.0077     |
| famrel   | 0.0059     |
| Dalc     | 0.0057     |
| Walc     | 0.0054     |

---

## Conclusion

The **Random Forest Regressor** achieved the best performance with an R² value of **0.9217**, explaining over 92% of the variance in the final grade (G3).

Key insights:

* The **second period grade (G2)** was the most influential predictor.
* **Absences** and **first period grade (G1)** were also significant.
* Behavioral aspects like **free time**, **social activity**, and **prior failures** contributed less but were still meaningful.

This suggests that **past academic performance** is the most reliable indicator of future outcomes.
