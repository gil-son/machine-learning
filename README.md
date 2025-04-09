# 🧠 Types of Machine Learning

## 💡 1. Supervised Learning
This is by far the most widely used type of ML in real-world applications.

- **What it is:** You train a model on labeled data (i.e., the input and expected output are both known).
- **Use Cases:**
  - Email spam detection
  - Credit scoring
  - Medical diagnosis
  - House price prediction

### ✅ Popular Algorithms

<details>
  <summary>📊 Linear Regression</summary>

- **Concept:** Predicts a continuous value (e.g., student test score) based on one or more input features.
- **Essential Math:**
  
    # $y = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b$


- It minimizes the **Mean Squared Error (MSE)** between predicted and actual values.
- **Use Case:** Predicting prices, trends, or scores.

</details>

<details>
  <summary>🧮 Logistic Regression</summary>

- **Concept:** Used for binary classification (e.g., pass/fail, spam/ham).
- **Essential Math:**

  # $P(y = 1 \mid x) = \sigma(w_1x_1 + w_2x_2 + \cdots + w_nx_n + b)$

  Where the **sigmoid function** is:

  # $\sigma(z) = \frac{1}{1 + e^{-z}}$

- **Use Case:** Disease prediction, marketing response, fraud detection.

</details>

<details>
  <summary>🌳 Decision Trees</summary>

- **Concept:** A flowchart-like structure where each internal node splits the data based on a feature.
- **Essential Math:**

  - **Gini Impurity:**
    # $G = 1 - \sum_{i=1}^{C} p_i^2$
    
  - **Entropy (for Information Gain):**
    # $H = - \sum_{i=1}^{C} p_i \log_2(p_i)$

- **Use Case:** Customer segmentation, credit risk modeling.

</details>

<details>
  <summary>🌲 Random Forest</summary>

- **Concept:** An ensemble of decision trees trained on random subsets of data and features.
- **Essential Math:**
  - For **Regression**:

    # ŷ = (1 / T) × (y₁ + y₂ + ... + yₜ)

- For **Classification**:

   # ŷ = majority vote of (y₁, y₂, ..., yₜ)

- **Use Case:** Robust classification and regression tasks, e.g., loan approval, stock prediction.

</details>

<details>
  <summary>📐 Support Vector Machines (SVM)</summary>

- **Concept:**
  - Finds the hyperplane that best separates the data into classes.
- **Essential Math:**
  - Decision boundary:
    # $w \cdot x + b = 0$
  - Optimization constraint:
    # $y_i(w \cdot x_i + b) \geq 1$
  - Margin to maximize:
    # $\frac{2}{\lVert w \rVert}$
- **Can use the _kernel trick_** (e.g., RBF kernel) to handle **non-linear** decision boundaries.  
- **Use Case:** Text classification, face recognition, bioinformatics.

</details>

<details>
  <summary>👥 k-Nearest Neighbors (kNN)</summary>

- **Concept:** Classifies a sample based on the majority vote (classification) or average (regression) of its k closest neighbors.
- **Essential Math:**

  - **Euclidean Distance:**
    # $d(x, x') = \sqrt{ \sum_{i=1}^{n} (x_i - x'_i)^2 }$

- **Other distance metrics** can be used, such as **Manhattan**, **Cosine**, or **Minkowski**, depending on the data.
- **Use Case:** Recommender systems, image classification, anomaly detection.

</details>

---

## 🧠 2. Unsupervised Learning

- **What it is:** The model tries to find patterns and groupings in the data without labeled outputs.
- **Use Cases:**
  - Customer segmentation
  - Market basket analysis
  - Anomaly detection
- **Popular Algorithms:**
  - k-Means Clustering
  - DBSCAN
  - PCA (Principal Component Analysis)
- **Python Libraries:** `scikit-learn`, `scipy`, `matplotlib`

---

## 🤖 3. Reinforcement Learning

- **What it is:** An agent learns to make decisions by interacting with an environment and getting feedback (rewards or penalties).
- **Use Cases:**
  - Robotics
  - Game playing (e.g., AlphaGo)
  - Self-driving cars
- **Popular Libraries:** `OpenAI Gym`, `Stable-Baselines`, `TensorFlow`, `PyTorch`

---

## 🧪 4. Semi-Supervised Learning

- **What it is:** Combines a small amount of labeled data with a large amount of unlabeled data to improve learning when labeling is expensive.
- **Use Cases:**
  - Web page classification
  - Medical imaging
  - Speech recognition
  - Fraud detection
- **Popular Algorithms:**
  - Self-training
  - Label propagation
  - Semi-supervised Support Vector Machines (S3VM)
  - Graph-based methods
- **Python Libraries:** `scikit-learn`, `sklearn.semi_supervised`, `TensorFlow`, `PyTorch`

