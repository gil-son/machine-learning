# Evaluation

## <img src="https://cdn-icons-png.flaticon.com/512/8592/8592294.png" width="80"/>  Overview


Model evaluation is the process of measuring how well a machine learning model performs on unseen data.  
It helps determine whether a model is **accurate**, **generalizable**, and **ready for production**.

Evaluation is typically divided into two main components:

- **Metrics** → How good is the model?
- **Validation strategies** → How reliable is the measurement?

---

## <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/5557/5557844.png" width="80"/> Why Evaluation Matters

Without proper evaluation:

- A model may **overfit** training data  
- Performance may look **better than reality**  
- Models cannot be **compared fairly**  
- Production performance may **degrade**  

Evaluation ensures the model generalizes to **real-world data**.

---

## <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/7527/7527144.png" width="80"/>  Evaluation Pipeline

```
Dataset
   ↓
Train / Validation Split
   ↓
Model Training
   ↓
Predictions
   ↓
Evaluation Metrics
   ↓
Model Selection
```

---

## <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/6061/6061551.png" width="80"/> Types of Evaluation Metrics <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/5567/5567532.png" width="80"/>

### Classification Metrics
Used when predicting **categories**

Examples:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- ROC-AUC  
- Log Loss  

Used for:

- Spam detection  
- Fraud detection  
- Image classification  
- Sentiment analysis  

---

### Regression Metrics
Used when predicting **continuous values**

Examples:

- MAE (Mean Absolute Error)  
- MSE (Mean Squared Error)  
- RMSE  
- R² Score  
- MAPE  

Used for:

- House price prediction  
- Sales forecasting  
- Demand prediction  
- Temperature prediction  

---

### Ranking / Retrieval Metrics
Used for **search, recommendation, and RAG systems**

Examples:

- Precision@K  
- Recall@K  
- MAP  
- NDCG  
- Hit Rate  

Used for:

- Search engines  
- Recommender systems  
- LLM retrieval (RAG)  
- Ranking problems  

---

## <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/3193/3193565.png" width="80"/> Validation Strategies

Evaluation metrics must be computed on **unseen data**.  
Validation strategies define how the data is split.

Common approaches:

- Train / Test Split  
- Cross Validation  
- K-Fold Cross Validation  
- Stratified K-Fold  
- Leave One Out  
- Time Series Split  
- Bootstrap  

---

## <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/5567/5567532.png" width="80"/> Choosing the Right Metric

Different problems require different metrics:

| Problem Type | Recommended Metrics |
|-------------|---------------------|
| Balanced classification | Accuracy |
| Imbalanced classification | F1, Precision, Recall |
| Regression | MAE, RMSE |
| Ranking | Precision@K, NDCG |
| Probabilities | Log Loss, ROC-AUC |

---

## <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/5567/5567532.png" width="80"/> Metrics vs Validation <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/7444/7444392.png" width="80"/>

| Concept | Purpose |
|--------|--------|
| Metric | Measures model performance |
| Validation | Ensures fair measurement |
| Test Set | Final unbiased evaluation |

Both are required for proper model evaluation.
