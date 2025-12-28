<h1 align="center">💳 Credit Card Fraud Detection (Imbalanced Classification)</h1>

## Overview
This project builds and evaluates multiple supervised machine learning models for **credit card fraud detection** on a highly imbalanced dataset. The goal is to correctly classify transactions as **fraudulent (1)** or **genuine (0)** and compare models using **error analysis** (false positives vs false negatives), ROC-AUC, and Precision–Recall behavior.

## Dataset
- **Rows:** 284,807 transactions  
- **Fraud cases (Class=1):** 492 (**0.172%**)  
- **Genuine cases (Class=0):** 284,315  
- **Features:** 28 anonymized **PCA components** (`V1`–`V28`) + `Time` + `Amount`  
- **Label:** `Class` (1 = fraud, 0 = genuine)

This extreme class imbalance makes **accuracy misleading**, so we focus on:
- **False Negatives (FN):** fraud missed (high cost)
- **False Positives (FP):** genuine flagged as fraud (operational burden)
- **Recall/Precision for fraud class**
- **ROC-AUC and Average Precision (AP)**

---

## Workflow (Process Summary)
1. **Load data** and confirm structure, missing values, and class imbalance.
2. **Split data** into train/test with **stratification** (preserve fraud rate).
3. Train and evaluate these models:
   - Logistic Regression (class_weight=balanced)
   - Random Forest (class_weight=balanced)
   - LightGBM (used as “XGBoost-style boosting” in the notebook)
   - Gradient Boosting (implemented with `LGBMClassifier` in this notebook)
4. **Evaluate** using:
   - Confusion Matrix
   - Classification Report (precision/recall/f1)
   - ROC Curve + AUC
   - Precision–Recall Curve + AP
5. Select the **best model** based on **balanced error trade-off** (low FN without exploding FP).

---

## Models and Results

### 1) Logistic Regression (balanced)
**Classification (Fraud=1):**
- Precision: **0.07**
- Recall: **0.88**
- Confusion Matrix:  
  - TN=83,481 | FP=1,814  
  - FN=18     | TP=130  
**AUC:** 0.9679  

**Error analysis:** Catches most fraud (low FN) but generates massive false alarms (very high FP).

---

### 2) Random Forest (balanced)
**Classification (Fraud=1):**
- Precision: **0.97**
- Recall: **0.70**
- Confusion Matrix:  
  - TN=85,292 | FP=3  
  - FN=44     | TP=104  
**AUC:** 0.9275  

**Error analysis:** Very low false alarms, but misses too many fraud cases (higher FN).

---

### 3) XGBoost-style boosting in notebook
**Classification (Fraud=1):**
- Precision: **0.86**
- Recall: **0.80**
- Confusion Matrix:  
  - TN=85,276 | FP=19  
  - FN=29     | TP=119  
**AUC:** 0.9697 | **AP:** ~0.83  

**Error analysis:** Strong balance—few false alarms and fewer missed fraud compared to Random Forest.

---

### 4) Gradient Boosting (implemented with `LGBMClassifier`)
**Classification (Fraud=1):**
- Precision: **0.86**
- Recall: **0.80**
- Confusion Matrix:  
  - TN=85,276 | FP=19  
  - FN=29     | TP=119  
**AUC:** 0.9697 | **AP:** ~0.83  

**Note:** In this notebook, “Gradient Boosting” uses **LightGBM’s `LGBMClassifier`**, so results match the LightGBM section.

---

## Best Model (Recommendation)
**LightGBM / Boosted Trees (GBDT family)** is the best choice here.

**Why:** It offers the best operational trade-off:
- **Low FN (missed fraud)** compared to Random Forest
- **Low FP (false alarms)** compared to Logistic Regression
- Strong ranking performance (**AUC ~ 0.97, AP ~ 0.83**)  
This makes it more suitable for real fraud pipelines where both missed fraud and excessive alerts are costly.

---

## Gradient Boosting Error Analysis


### Confusion Matrix (Gradient Boosting)
![Confusion Matrix - Gradient Boosting](Confusion%20Matrix%20Output.png)

**Discussion:** Only **19 FP** and **29 FN**, meaning low false alarms while still catching most fraud.

### ROC Curve (Gradient Boosting)
![ROC Curve - Gradient Boosting](ROC%20Curve.png)

**Discussion:** High separability (AUC ≈ **0.97**) shows strong ranking ability.

### Precision–Recall Curve (Gradient Boosting)
![Precision-Recall Curve - Gradient Boosting](Precision%20Recall%20Curve.png)

**Discussion:** AP ≈ **0.83** indicates strong fraud detection performance under class imbalance.

---

## Reproducibility

### Environment Setup
```bash
# create environment (optional)
conda create -n fraud-detection python=3.10 -y
conda activate fraud-detection

# install dependencies
pip install -U pandas numpy matplotlib seaborn scikit-learn lightgbm
````

### Run

* Open the notebook / script and run cells top-to-bottom.
* Ensure dataset path matches:

  * `datasets/credit_card_data/credit_card.csv`

---

## Project Structure (Suggested)

```
credit-card-fraud-detection/
│
├─ datasets/
│  └─ credit_card_data/
│     └─ credit_card.csv
│
├─ credit-card-fraud-detection.ipynb
│
├─ figures/
│     ├─ Confusion Matrix Output.png
│     ├─ ROC Curve.png
│     └─ Precision Recall Curve.png
│
└─ README.md
```

---

## Reference 

**Hands-on Unsupervised Learning Using Python**

Author: **Ankur A. Patel**

A practical guide to applying unsupervised learning to uncover patterns in unlabeled data using production-ready Python tools (scikit-learn and TensorFlow), covering anomaly detection, feature engineering, synthetic data generation, and more.

Official Book Website: https://www.unsupervisedlearningbook.com/thebook

Available on Amazon: https://www.amazon.com/Hands-Unsupervised-Learning-Using-Python/dp/1492035645

Available on O'Reilly Safari: https://www.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/

More on the Author: https://www.ankurapatel.io


