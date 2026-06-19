### <img src="https://i.ibb.co/676KwYXF/random-forest.png" width="70"/> Random Forests: The Wisdom of the Crowd

---

## 🌳 From Single Trees to a Forest

We've explored [Decision Trees](https://github.com/gil-son/machine-learning/blob/main/algorithms/decision_trees.md). Now, let's delve into **Random Forests**, an ensemble method using multiple decision trees for robust predictions by combining their insights.

---

## 🤝 What is a Random Forest?

A **Random Forest** is a **supervised learning algorithm** for **classification** and **regression**. It trains multiple decision trees and outputs the mode (classification) or mean/average (regression) of their predictions.

Key ideas:

-   **Ensemble Learning:** Combines multiple "weak" decision trees into a "stronger" learner.
-   **Bagging (Bootstrap Aggregating):** Each tree trains on a random subset of the original data (sampled with replacement), increasing training set diversity.
-   **Feature Randomness:** Only a random subset of features is considered at each node split, further decorrelating trees.

---

## 🛠️ How a Random Forest is Built

Analogy: Predicting rain by consulting multiple weather experts with slightly different data and focus.

1.  **Bootstrap Sampling:** Create multiple training datasets by randomly sampling the original data with replacement.
2.  **Tree Building with Feature Randomness:** Build a decision tree for each bootstrapped dataset. At each node split, consider only a random subset of features to find the best split.
3.  **No Pruning (Typically):** Individual trees are grown deep without pruning; the ensemble effect mitigates overfitting.

---

## 🔮 Making Predictions

-   **Classification:** The final prediction ($\hat{y}$) is the class with the most votes from all $T$ trees ($y_1, y_2, ..., y_T$).
    $$\hat{y} = \text{majority vote of } (y_1, y_2, ..., y_T)$$
-   **Regression:** The final prediction ($\hat{y}$) is the average of the predictions from all $T$ trees ($y_1, y_2, ..., y_T$).
    $$\hat{y} = \frac{1}{T} \times (y_1 + y_2 + ... + y_T)$$

---

## 🌟 Why Does it Work So Well?

-   **Reduced Variance (Bagging):** Training on different data subsets makes trees learn diverse patterns, reducing the impact of outliers and noise, leading to a more stable model.
-   **Reduced Correlation (Feature Randomness):** Considering only a feature subset at each split decorrelates trees, forcing them to consider different data aspects and creating a more diverse ensemble.

---

## ⚙️ Key Hyperparameters

-   `n_estimators`: Number of trees (more generally improves performance but increases cost).
-   `max_features`: Number of features considered at each split (smaller reduces tree correlation; common values: $\sqrt{M}$ for classification, $M$ for regression).
-   `max_depth`: Maximum depth of individual trees (can limit depth).
-   `min_samples_split`: Minimum samples to split an internal node.
-   `min_samples_leaf`: Minimum samples at a leaf node.

---

## 📊 Feature Importance

Estimates feature importance by measuring the decrease in prediction accuracy when a feature is randomly permuted. Larger decreases indicate higher importance.

---

## ✅ Pros & Cons (Compared to Single Decision Trees)

### ✅ Pros:

-   Higher accuracy.
-   Reduced overfitting.
-   Robust to outliers.
-   Handles high dimensionality.
-   Provides feature importance.

### ❌ Cons:

-   Less interpretable (black box).
-   Higher computational cost.

---

## 📦 Used In:

-   Image classification
-   Object detection
-   Natural language processing
-   Financial modeling
-   Bioinformatics
-   Recommender systems

---

## 🚀 Stepping Beyond: Further Explorations

Explore other ensemble methods like **Gradient Boosting Machines** or delve deeper into ensemble learning theory.

---

## ⚙️ Core Algorithms in Action

While the high-level concept of a Random Forest is consistent, the underlying implementation involves specific algorithms for tree construction and aggregation. Here are the main steps and algorithms at play:

1.  **Bootstrap Sampling (Bagging):** Create $k$ training sets of size $N$ by randomly sampling the original $N$ instances with replacement.
2.  **Feature Subset Selection:** At each node split, randomly select $m < M$ features to consider for the best split.
3.  **Decision Tree Induction (with Random Subspace):** Grow a decision tree on each bootstrapped dataset, considering only the random feature subset at each split, typically without pruning.

---

## 🌍 Real-World Scenario Examples

Let's explore how Random Forests are applied in different domains:

**1. E-commerce: Customer Churn Prediction**

* **Scenario:** Identify likely churners.
* **Data:** Demographics, purchase history, activity; Target: churned (yes/no).
* **Random Forest Application:** Classifier.
* **Benefit:** Proactive retention; churn indicators.
* **Calculation Insight:** Tree building uses impurity measures (Gini/Entropy) for splits. Prediction is the majority class vote.

    <p>Imagine a split on 'Purchase Frequency' leading to child nodes with different churn rates. The Gini impurity would be calculated for each node to determine the best split. The final churn prediction for a customer is based on the majority vote of individual trees.</p>

**2. Healthcare: Disease Diagnosis**

* **Scenario:** Predict disease presence.
* **Data:** Tests, symptoms; Target: present (yes/no).
* **Random Forest Application:** Classifier.
* **Benefit:** Accurate diagnoses; key features.
* **Calculation Insight:** Each tree votes on the diagnosis. The final prediction is the class with the majority of votes.

    <p>If a Random Forest with 500 trees is used, and for a new patient, 420 trees predict "Disease Present," then that would be the final prediction.</p>

**3. Finance: Credit Risk Assessment**

* **Scenario:** Assess loan applicant creditworthiness.
* **Data:** Credit score, income, debt; Target: default (yes/no).
* **Random Forest Application:** Classifier.
* **Benefit:** Informed lending; risk factors.
* **Calculation Insight:** Majority vote on default prediction across trees.

**4. Environmental Science: Species Distribution Modeling**

* **Scenario:** Predict species distribution.
* **Data:** Observation locations, environmental variables; Target: presence/absence (or abundance).
* **Random Forest Application:** Classifier/Regressor.
* **Benefit:** Ecological understanding; climate change impact; conservation.
* **Calculation Insight:** Majority vote for presence/absence; average of predicted abundance values for regression.

    <p>If 100 trees predict the number of plants in a region, the final prediction is the average of these 100 predicted values.</p>

These "Calculation Insight" snippets provide a glimpse into the type of computations involved at key stages within the Random Forest algorithm for each scenario, without requiring detailed numerical examples on large datasets.
