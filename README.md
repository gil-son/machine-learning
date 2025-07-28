# <img src="https://cdn-icons-png.flaticon.com/512/6062/6062189.png" width="80"/> Types of Machine Learning

## <img src="https://cdn-icons-png.flaticon.com/512/6229/6229938.png" width="70"/> Supervised Learning
This is by far the most widely used type of ML in real-world applications.

- **What it is:** You train a model on labeled data (i.e., the input and expected output are both known).
- **Use Cases:**
  - Email spam detection
  - Credit scoring
  - Medical diagnosis
  - House price prediction

### ✅ Popular Algorithms

<details>
  <summary><img src="https://cdn-icons-png.flaticon.com/512/2620/2620536.png" width="50"/> Linear Regression</summary>

- **Concept:** Predicts a continuous value (e.g., student test score) based on one or more input features.
- **Essential Math:**
  
    # $y = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b$


- It minimizes the **Mean Squared Error (MSE)** between predicted and actual values.
- **Use Case:** Predicting prices, trends, or scores.

</details>

<details>
  <summary><img src="https://cdn-icons-png.flaticon.com/512/3295/3295481.png" width="50"/> Logistic Regression</summary>

- **Concept:** Used for binary classification (e.g., pass/fail, spam/ham).
- **Essential Math:**

  # $P(y = 1 \mid x) = \sigma(w_1x_1 + w_2x_2 + \cdots + w_nx_n + b)$

  Where the **sigmoid function** is:

  # $\sigma(z) = \frac{1}{1 + e^{-z}}$

- **Use Case:** Disease prediction, marketing response, fraud detection.

</details>

<details>
  <summary><img src="https://cdn-icons-png.flaticon.com/512/1960/1960357.png" width="50"/> Decision Trees</summary>

- **Concept:** A flowchart-like structure where each internal node splits the data based on a feature.
- **Essential Math:**

  - **Gini Impurity:**
    # $G = 1 - \sum_{i=1}^{C} p_i^2$
    
  - **Entropy (for Information Gain):**
    # $H = - \sum_{i=1}^{C} p_i \log_2(p_i)$

- **Use Case:** Customer segmentation, credit risk modeling.

</details>

<details>
  <summary><img src="https://i.ibb.co/676KwYXF/random-forest.png" width="50"/> Random Forest</summary>

- **Concept:** An ensemble of decision trees trained on random subsets of data and features.
- **Essential Math:**
  - For **Regression**:

    # ŷ = (1 / T) × (y₁ + y₂ + ... + yₜ)

- For **Classification**:

   # ŷ = majority vote of (y₁, y₂, ..., yₜ)

- **Use Case:** Robust classification and regression tasks, e.g., loan approval, stock prediction.

</details>

<details>
  <summary><img src="https://i.ibb.co/4R3pTJyj/svm.png" width="50"/>  Support Vector Machines (SVM)</summary>

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
  <summary><img src="https://i.ibb.co/MkS0BttC/knn.png" width="50"/> k-Nearest Neighbors (kNN)</summary>

- **Concept:** Classifies a sample based on the majority vote (classification) or average (regression) of its k closest neighbors.
- **Essential Math:**

  - **Euclidean Distance:**
    # $d(x, x') = \sqrt{ \sum_{i=1}^{n} (x_i - x'_i)^2 }$

- **Other distance metrics** can be used, such as **Manhattan**, **Cosine**, or **Minkowski**, depending on the data.
- **Use Case:** Recommender systems, image classification, anomaly detection.

</details>

---

## <img src="https://cdn-icons-png.flaticon.com/512/6062/6062161.png" width="60"/> Unsupervised Learning

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

## <img src="https://cdn-icons-png.flaticon.com/512/10087/10087719.png" width="60"/> Reinforcement Learning

- **What it is:** An agent learns to make decisions by interacting with an environment and getting feedback (rewards or penalties).
- **Use Cases:**
  - Robotics
  - Game playing (e.g., AlphaGo)
  - Self-driving cars
- **Popular Libraries:** `OpenAI Gym`, `Stable-Baselines`, `TensorFlow`, `PyTorch`

---

## <img src="https://cdn-icons-png.flaticon.com/512/1713/1713891.png" width="60"/> Semi-Supervised Learning

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

<hr/>

# <img src="https://cdn-icons-png.flaticon.com/512/6062/6062189.png" width="80"/> Machine Learning Techniques

- **Classification**  
  A supervised learning task where the model learns to categorize data into predefined **classes or labels**.  
  **Example:** Predicting if an email is *spam* or *not spam*.

- **Regression**  
  A supervised learning task where the goal is to predict a **continuous value**.  
  **Example:** Predicting the **price of a house** based on size, location, etc.

- **Clustering**  
  An **unsupervised learning** method where the algorithm groups data into **clusters** based on similarity—without predefined labels.  
  **Example:** Segmenting customers into groups based on their behavior or purchases.

- **Anomaly Detection**  
  Identifying data points that are **unusual or deviate** significantly from the majority.  
  **Example:** Detecting **fraudulent credit card transactions**.

- **Sequence Mining**  
  Analyzing and identifying **patterns in ordered data** (sequences), especially over time.  
  **Example:** Finding common sequences in **customer purchases** or website navigation.

- **Dimension Reduction**  
  Reducing the number of features (dimensions) in a dataset while keeping important information—used to simplify models and visualize high-dimensional data.  
  **Example:** Using **PCA (Principal Component Analysis)** to reduce image data with thousands of pixels into just a few features.

- **Recommendation System**  
  A system that suggests **items** (movies, products, etc.) to users based on their preferences or behaviors.  
  **Example:** Netflix recommending **movies or shows** based on your watch history.

<hr/>

# <img src="https://cdn-icons-png.flaticon.com/512/6062/6062189.png" width="80"/> Machine Learning Model Lifecycle

- **Problem Definition**  
  Clearly define the **objective** of the machine learning task.  
  **Example:** Predict customer churn or classify product reviews as positive or negative.

- **Data Collection**  
  Gather relevant and sufficient **raw data** from various sources like databases, APIs, sensors, or manual input.  
  **Example:** Collecting user behavior logs or survey results.

- **Data Preparation**  
  Clean, transform, and structure the data for training. This includes **handling missing values**, **encoding categories**, and **normalizing** values.  
  **Example:** Converting text into numeric form or removing outliers.

- **Model Development and Evaluation**  
  Choose a model type, train it using prepared data, and evaluate its **accuracy, precision, recall**, or other relevant metrics.  
  **Example:** Training a decision tree and evaluating it using cross-validation.

- **Model Deployment**  
  Integrate the trained model into a **production environment** where it can receive real input and make predictions.  
  **Example:** Deploying a fraud detection model via an API to monitor real-time transactions.


<hr/>
<div align="center">
  <img src="https://i.ibb.co/kgNSnpv/git-support.png">
</div>


