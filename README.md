# 🧠 Multimodel Time Series Classification with CNN, LSTM, and XGBoost

This project performs binary classification on time series data using an ensemble of deep learning and classical machine learning models. The goal is to optimize classification performance — particularly the F1 Score — by combining outputs from a CNN, LSTM, and XGBoost through a fusion and threshold tuning strategy.

---

## 📂 Project Structure

```
.
├── data/
│   └── (Preprocessed input features)
├── models/
│   ├── best_model_cnn.pth
│   └── best_model_lstm.pth
├── outputs/
│   ├── probs_cnn.pkl
│   ├── probs_lstm.pkl
│   ├── probs_xgb.pkl
│   └── fusion_weights.pkl
├── scripts/
│   ├── train_cnn.py
│   ├── train_lstm.py
│   ├── train_xgb.py
│   ├── optimize_fusion.py
│   └── utils.py
└── README.md
```

---

## 🔧 Models Used

### 1. **Convolutional Neural Network (CNN)**
- 1D convolution on multivariate time series
- Trained using Focal Loss to handle class imbalance
- Validation probabilities saved to `probs_cnn.pkl`

### 2. **LSTM**
- Sequence-based model for capturing temporal dependencies
- Returns class probabilities saved as `probs_lstm.pkl`

### 3. **XGBoost**
- Gradient boosting classifier using summary/statistical features
- Returns output probabilities to `probs_xgb.pkl`

---

## ♻️ Fusion Strategy

Model outputs are fused using a weighted sum:

\[
\text{fused\_probs} = w_1 \cdot \text{probs\_lstm} + w_2 \cdot \text{probs\_cnn} + w_3 \cdot \text{probs\_xgb}
\]

- **Weights (`w1`, `w2`, `w3`)** and **classification threshold** are jointly optimized to maximize **F1 Score**.
- Optimization is performed using `scipy.optimize` with constraints:
  - All weights ≥ 0.1
  - Weights sum to 1.0
  - Threshold ∈ [0.1, 0.9]

---

## 📈 Evaluation Metrics

Evaluated on the validation set using:

- **Accuracy**
- **F1 Score**
- **Precision & Recall**
- **AUC (Area Under ROC Curve)**
- **Balanced Accuracy**
- **Confusion Matrix**
- **ROC and F1-Threshold plots**

---

## 🚀 How to Run

1. **Preprocess your data** into `X_train_ready`, `X_val`, `y_train`, `y_val`
2. **Train each model:**
   ```bash
   python scripts/train_cnn.py
   python scripts/train_lstm.py
   python scripts/train_xgb.py
   ```
3. **Run fusion optimization:**
   ```bash
   python scripts/optimize_fusion.py
   ```

---

## 📌 Requirements

- Python ≥ 3.8
- PyTorch
- scikit-learn
- numpy
- tqdm
- matplotlib, seaborn
- xgboost
- scipy

Install via:
```bash
pip install -r requirements.txt
```

---

## 🤝 Acknowledgements

- Inspired by best practices in ensemble learning and hybrid deep learning models for imbalanced classification.
- CNN and LSTM models incorporate techniques like **Focal Loss** and **early stopping** to improve robustness.

---

## 📬 Contact

If you have questions or suggestions, feel free to open an issue or reach out.

---
