# Fake News Detection

![image](https://thumbs.dreamstime.com/b/fake-news-word-cloud-white-background-fake-news-word-cloud-102179857.jpg)

## Overview

This project explores fake news detection with different techniques, heuristic, classical machine learning and modern deep learning. The goal is to benchmark different models, assess their robustness across real and synthetic data, and analyze misclassifications using explainability tools.

---

## Project Goals

- Build and compare ML and DL models for fake news classification.
- Use preprocessing strategies like lemmatization and entity masking.
- Leverage LIME and attention visualization for model interpretability.
- Evaluate models on both original and synthetically generated datasets.

---

## Dataset Description

The dataset merges real and fake news articles, with balanced labels and cleaned text. A synthetic set was generated to simulate different writing styles and test generalizability.

Challenges include dealing with politically ambiguous language, clickbaiting titles and varying writing tone or structure between real and fake news.

---

## Modeling Approach

### Heuristic
- Feature extraction and engineering.

### Machine Learning
- TF-IDF vectorization combined with numeric features.
- SVD used for dimensionality reduction.
- Logistic Regression, Random Forest, LightGBM, SVM evaluated with Optuna tuning.
- A simple heuristic model tested based on keyword presence.

### Deep Learning
- Fine-tuned DistilBERT model on masked and lemmatized text.
- Training done using PyTorch Lightning with evaluation callbacks and metrics logging.

---

## Data Preprocessing

- Cleaned and tokenized text.
- Masked named entities (e.g., _person_, _organization_) to reduce overfitting.
- Lemmatization and stopword removal to standardize input.
- Applied to both real and synthetic datasets for consistency.

---

## Conclusion

This project demonstrates the value of deep learning in fake news detection, especially with pre-trained transformers like DistilBERT. While simple models like Logistic Regression offer competitive results on clean, structured data, DistilBERT excels in handling noisy, diverse, and synthetic inputs. Explainability techniques reveal key patterns behind predictions and help debug model behavior. This work can support real-world applications like news filtering tools or fact-checking assistants, where speed and accuracy trade-offs matter. Challenges may arise when applying this to non-political domains or when adversarially generated content mimics real writing styles.

---

## How to Run the Project

1. Clone the repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run notebooks in order: `fakenews-eda.ipynb`, `fakenews-ml.ipynb`, `fakenews-dl.ipynb`

---

## Future Work

- Support multisubjects datasets.
- Deploy real-time classification via API.
- Expand interpretability using SHAP and advanced attention visualization.

---

## License

This project is licensed under the MIT License.