# Fake News Detection (Technical Test)

A full workflow for detecting fake vs. real news articles using the Kaggle Fake-and-Real News dataset, with data augmentation from FakeNewsNet. This project demonstrates robust modeling, feature engineering, and analysis on modern NLP tabular/text tasks.

---

## 📦 Data Sources

**Primary:**  
- [Kaggle Fake-and-Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news)  
  - Files: `True.csv`, `Fake.csv` (news articles with title, text, domain, date, label)

**Auxiliary:**  
- [FakeNewsNet Minimal](https://github.com/KaiDMML/FakeNewsNet)  
  - Files: `politifact_fake.csv`, `politifact_real.csv`, `gossipcop_fake.csv`, `gossipcop_real.csv` (news headlines/labels)

---

## 🚀 Quick Start

1. **Clone this repo:**
    ```bash
    git clone <your-github-url>
    cd <project-folder>
    ```

2. **Install requirements:**  
    ```bash
    pip install -r requirements.txt
    ```

3. **Download datasets:**
    - Place `True.csv` and `Fake.csv` from Kaggle in the `/data` folder.
    - Place FakeNewsNet minimal CSVs in the same `/data` folder.  
      *(Scripts provided in the notebook will attempt to fetch them automatically if missing.)*

4. **Run the Jupyter Notebook:**  
    - Open `FakeNewsDetection.ipynb`  
    - Execute cells sequentially for EDA, feature engineering, modeling, and evaluation.

---

## 📝 Project Structure
.
├── data/ # All CSVs here (Kaggle + FakeNewsNet)
├── notebooks/
│ └── FakeNewsDetection.ipynb # Main analysis notebook
├── requirements.txt
└── README.md


---

## 🧠 Workflow Overview

### **1. Data Preparation**
- Combine and clean Kaggle and FakeNewsNet datasets.
- Remove potential metadata leaks (e.g., publisher tags, weekday names).

### **2. Exploratory Data Analysis**
- Visualize class and domain distributions.
- Compare text length, top tokens, and data variety.

### **3. Baseline Model**
- **Logistic Regression** on TF–IDF + length features.
- *Initial leakage detected* (domain & metadata features), so these are dropped/pruned.

### **4. Advanced Feature Engineering**
- Remove metadata artifacts (e.g., “reuters”, “via”, “monday”, etc.).
- Add VADER sentiment, Flesch–Kincaid, and Gunning Fog readability scores.

### **5. Data Augmentation**
- Integrate FakeNewsNet minimal dataset (Politifact + GossipCop) to improve generalization and robustness.
- Re-apply all feature engineering steps.

### **6. Model Architecture Exploration**
- **XGBoost Classifier:** Nonlinear modeling on engineered features (TF–IDF + numeric).
- **Transformer Approach:** (Deferred) Planned to fine-tune DistilBERT for comparison, but left for future work due to compute/time constraints.

---

## 🏆 Results & Key Choices

| Phase               | Model                | Fake F1 | Real F1 | Accuracy | ROC AUC | Notes                                      |
|---------------------|---------------------|---------|---------|----------|---------|--------------------------------------------|
| Baseline (leaky)    | Logistic Regression | 0.99    | 0.99    | 0.99     | —       | Inflated by metadata/domain leakage        |
| Cleaned             | Logistic Regression | 0.97    | 0.97    | 0.97     | —       | Metadata pruned, realistic performance     |
| +FakeNewsNet        | Logistic Regression | 0.90    | 0.93    | 0.92     | —       | Data richer, generalization improved       |
| +XGBoost            | XGBoost             | 0.90    | 0.93    | 0.92     | 0.97    | Best compromise of recall & speed          |
| Transformer (future)| DistilBERT          | —       | —       | —        | —       | Will be compared in future work            |

- **Final selection:** **XGBoost** on the augmented dataset as the best performing tabular model given time and compute.

---

## 🔍 Analysis & Interpretation

- High scores on a single-source dataset are often due to data leakage (e.g., unique domains or publisher artifacts).
- Data augmentation with FakeNewsNet, combined with feature pruning, provides a realistic and robust benchmark.
- XGBoost outperformed logistic regression on challenging (nonlinear) text features, while remaining interpretable and fast.

---

## 📝 Methodological Notes

- **No test data leakage:** All splits stratified by label, no metadata artifacts in features.
- **All results are fully reproducible** via the notebook.
- **Advanced transformer-based methods are planned as a future extension** (code scaffold and speed-up tips included).

---

## ⏭️ Future Work

- Fine-tune DistilBERT (or similar) for direct comparison with XGBoost.
- Hyperparameter tuning (Optuna, GridSearch) for further improvements.
- Ensemble approaches (e.g., stacking/soft-voting with LR, XGBoost, and transformer).
- Model explainability via SHAP/LIME.
- Add more news datasets for even stronger generalization.

---

## 🙏 Acknowledgements

- [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news)
- [FakeNewsNet Project](https://github.com/KaiDMML/FakeNewsNet)
- HuggingFace Transformers and Datasets libraries

---

*Project by Nikhil Gupta*
