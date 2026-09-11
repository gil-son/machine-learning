# Evaluation <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

<p align="center">
  <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-evaluation.png?ref_type=heads" width="100%">
</p>

## Overview <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-megaman-protoman-compare.png?ref_type=heads" width="12%">

Model evaluation is the process of measuring how well a machine learning model performs on unseen data.  
It helps determine whether a model is **accurate**, **generalizable**, and **ready for production**.

Evaluation is typically divided into two main components:

- **Metrics** → How good is the model?
- **Validation strategies** → How reliable is the measurement?

---

## Why Evaluation Matters <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

Without proper evaluation:

- A model may **overfit** training data  
- Performance may look **better than reality**  
- Models cannot be **compared fairly**  
- Production performance may **degrade**  

Evaluation ensures the model generalizes to **real-world data**.

## Types of Evaluation Metrics <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

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

### Clustering Metrics
Used when grouping **unlabeled data** (k-means, DBSCAN, hierarchical clustering)

Examples:

- Silhouette Score  
- Davies-Bouldin Index  
- Calinski-Harabasz Index  
- Dunn Index  
- Inertia / WCSS  

Used for:

- Customer segmentation  
- Anomaly detection  
- Document/topic grouping  

> These require no ground-truth labels — they judge cluster *quality* directly (how compact and well-separated clusters are) rather than comparing to a known answer.

---

### Dimensionality Reduction Metrics
Used for **PCA, t-SNE**, and similar techniques

Examples:

- Explained Variance Ratio  
- Reconstruction Error  
- Trustworthiness / Continuity  

Used for:

- Feature compression  
- Visualization of high-dimensional data  
- Noise reduction before downstream modeling  

---

### Reinforcement Learning Metrics
Used for **agents learning through interaction** (Q-learning, SARSA, DQN, policy gradient)

Examples:

- Cumulative / Average Reward  
- Episode Length  
- Sample Efficiency  
- Success Rate  
- Regret  

Used for:

- Game-playing agents  
- Robotics control  
- Resource allocation / scheduling policies  

---

### LLM Evaluation Metrics (Bonus)
Used for **large language models and RAG pipelines**

| Metric | What it checks |
|---|---|
| **Correctness** | Is the output factually/logically right relative to a ground truth or expected answer? |
| **Faithfulness** | Does the output avoid contradicting or fabricating beyond its source content? |
| **Relevance** | Does the output actually address the user's query, without drifting off-topic? |
| **Completeness** | Does the output cover everything the query/task needs, without omitting required parts? |
| **Groundedness** | Can claims in the output be traced back to supporting evidence/sources? |
| **Context Precision** | Of the chunks retrieved (RAG), how many were actually relevant? |
| **Context Recall** | Of the information needed to answer (RAG), how much was successfully retrieved? |
| **Answer Relevance** | Does the final generated answer directly target the original question, without padding? |

**How these fit together:**
- **Correctness** vs. **Faithfulness/Groundedness** ask different questions: correctness checks against external truth, faithfulness/groundedness checks against the given source — a hallucinated answer can occasionally be correct by coincidence, and a faithful answer can occasionally be wrong if the source itself is wrong.
- **Context Precision** and **Context Recall** evaluate the *retriever* half of a RAG pipeline.
- **Relevance**, **Completeness**, and **Answer Relevance** evaluate the *generator's* output against the query.
- Use them together, not in isolation — e.g., high Context Recall with low Faithfulness means the right information was retrieved but the model didn't use it properly; high Context Precision with low Context Recall means retrieval is clean but missing key facts.

Used for:

- Chatbots and assistants  
- Retrieval-Augmented Generation (RAG)  
- Summarization and generation tasks  

---

## Validation Strategies <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

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

> These strategies are built for supervised learning. Unsupervised evaluation typically doesn't hold out a labeled test set (there's nothing to predict against); RL is instead validated through repeated rollouts across multiple random seeds/environment instances rather than a data split.

---

## Choosing the Right Metric <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/5567/5567532.png" width="80"/> 

Different problems require different metrics:

| Problem Type | Recommended Metrics |
|-------------|---------------------|
| Balanced classification | Accuracy |
| Imbalanced classification | F1, Precision, Recall |
| Regression | MAE, RMSE |
| Ranking | Precision@K, NDCG |
| Probabilities | Log Loss, ROC-AUC |
| Clustering | Silhouette Score, Davies-Bouldin Index |
| Dimensionality Reduction | Explained Variance Ratio, Trustworthiness |
| Reinforcement Learning | Cumulative Reward, Success Rate |
| LLM / RAG | Faithfulness, Context Precision/Recall, Answer Relevance |

---

## <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/5567/5567532.png" width="80"/> Metrics vs Validation <td align="center"><img src="https://cdn-icons-png.flaticon.com/512/7444/7444392.png" width="80"/>

| Concept | Purpose |
|--------|--------|
| Metric | Measures model performance |
| Validation | Ensures fair measurement |
| Test Set | Final unbiased evaluation |

Both are required for proper model evaluation.
