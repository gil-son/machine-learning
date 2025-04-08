## Types of Machine Learning

### 💡 Supervised Learning
This is by far the most widely used type of ML in real-world applications.

  - What it is: You train a model on labeled data (i.e., the input and expected output are both known).

  - Use Cases:
    - Email spam detection
    - Credit scoring
    - Medical diagnosis
    - House price prediction

- Popular Algorithms:
    - Linear Regression
    - Logistic Regression
    - Decision Trees
    - Random Forest
    - Support Vector Machines (SVM)
    - k-Nearest Neighbors (kNN)
    - Gradient Boosting (like XGBoost, LightGBM)
  - Python Libraries: scikit-learn, pandas, numpy
 
### 🧠 2. Unsupervised Learning
Used when you have no labels, just raw data.

  - What it is: The model tries to find patterns and groupings in the data.
  - Use Cases:
    - Customer segmentation
    - Market basket analysis
    - Anomaly detection
  - Popular Algorithms:
    - k-Means Clustering
    - DBSCAN
    - PCA (Principal Component Analysis)
  - Python Libraries: scikit-learn, scipy, matplotlib (for visualization)

### 🤖 3. Reinforcement Learning
This is more complex and used in specialized fields.
  - What it is: An agent learns to make decisions by interacting with an environment and getting feedback (rewards or penalties).
  - Use Cases:
    - Robotics
    - Game playing (e.g., AlphaGo)
    - Self-driving cars
  - Popular Libraries: OpenAI Gym, Stable-Baselines, TensorFlow, PyTorch

## 🧪 4. Semi-supervised Learning
Sits between supervised and unsupervised learning.

  - What it is: You train a model on a small amount of labeled data and a large amount of unlabeled data. The idea is to leverage the unlabeled data to improve learning accuracy when labeling is expensive or time-consuming.
  - Use Cases:
    - Web page classification
    - Medical imaging (where labeling each image is costly)
    - Speech recognition
    - Fraud detection
  - Popular Algorithms:
    - Self-training
    - Label propagation
    - Semi-supervised Support Vector Machines (S3VM)
  - Graph-based methods

Python Libraries: scikit-learn, TensorFlow, PyTorch, sklearn.semi_supervised (for label propagation and label spreading)
