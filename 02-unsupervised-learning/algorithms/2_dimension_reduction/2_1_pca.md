# PCA — Principal Component Analysis <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

PCA (1 feature) | PCA (n features) | Explained Variance | Reconstruction

---

## What is PCA? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

PCA (Principal Component Analysis) is an **unsupervised learning algorithm** used to reduce the number of features in a dataset while retaining as much of the original variance — and therefore information — as possible.

At its core, PCA finds new axes called **principal components** that are linear combinations of the original features, ordered so that the first component captures the most variance, the second captures the most remaining variance, and so on. By projecting data onto the top k components, PCA compresses high-dimensional data into a lower-dimensional space where patterns become easier to visualize, models train faster, and the curse of dimensionality is reduced — with minimal information loss.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to find a new coordinate system — the principal components — that best describes the spread of the data, then project all points onto the top k components to reduce dimensionality. Four concepts are central:

- **Mean centering** — subtract the mean of each feature so the data is centred at the origin. This is required before computing covariance.
- **Covariance matrix** — a square matrix capturing how much each pair of features varies together. Its eigenstructure reveals the directions of maximum variance.
- **Eigenvectors (principal components)** — the directions in feature space along which the data varies the most. The eigenvector with the largest eigenvalue is PC1, the next is PC2, and so on. All PCs are orthogonal to each other.
- **Eigenvalues** — the amount of variance captured by each principal component. Used to compute the explained variance ratio and decide how many components to keep.

**Two key hyperparameters:**

```
n_components  — how many principal components to keep after projection
               (chosen by explained variance threshold, e.g. 95%)
```

**Projection formula:**

```
Z = X_centred @ W
```

**Reconstruction formula:**

```
X_approx = Z @ W^T + mean
```

Where `W` is the matrix whose columns are the top k eigenvectors, `Z` is the projected (low-dimensional) data, and `X_approx` is the approximation of the original data.

---

## PCA (1 feature)

PCA with a single feature has a trivial outcome: there is only one possible direction, so PC1 is the feature itself. The only operation PCA performs is **mean centering** — shifting the data so its mean is zero. This illustrates the foundation of PCA before extending to multiple dimensions.

**Mean centering (1D):**

```
x_centred_i = x_i - mean(x)
```

Also written as:

```
PC1 score_i = x_centred_i   (since the only eigenvector in 1D is 1)

Variance captured = Var(x_centred) = Var(x)  (100% — nothing to reduce)
```

Where:

- `x_i` — original value of data point i
- `mean(x)` — the mean of all values in the feature
- `x_centred_i` — the mean-centred value; this is also the PC1 score in 1D
- `Var(x)` — the variance of the feature; PCA aims to preserve this when projecting

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Centring Study Hours Before Dimensionality Reduction</summary>
  <br/>

  Imagine you have seven students and a single feature — hours studied. PCA on one feature demonstrates mean centering and variance, which are the building blocks for the multi-feature case.

  **Dataset (Hours Studied):**

  | Student | Hours (x) |
  |---------|-----------|
  | S1      | 2         |
  | S2      | 3         |
  | S3      | 4         |
  | S4      | 7         |
  | S5      | 8         |
  | S6      | 9         |
  | S7      | 5         |

  **Step 1 — Compute the mean:**

  ```
  mean(x) = (2 + 3 + 4 + 7 + 8 + 9 + 5) / 7 = 38 / 7 = 5.4286
  ```

  **Step 2 — Mean-centre each value:**

  | Student | x  | x − mean   | Centred value |
  |---------|----|------------|---------------|
  | S1      | 2  | 2 − 5.4286 | −3.4286       |
  | S2      | 3  | 3 − 5.4286 | −2.4286       |
  | S3      | 4  | 4 − 5.4286 | −1.4286       |
  | S4      | 7  | 7 − 5.4286 | +1.5714       |
  | S5      | 8  | 8 − 5.4286 | +2.5714       |
  | S6      | 9  | 9 − 5.4286 | +3.5714       |
  | S7      | 5  | 5 − 5.4286 | −0.4286       |

  **Step 3 — Compute variance (the eigenvalue in 1D):**

  ```
  Var(x) = mean( (x_i - mean)^2 )
         = ( 3.4286² + 2.4286² + 1.4286² + 1.5714² + 2.5714² + 3.5714² + 0.4286² ) / 7
         = ( 11.755 + 5.898 + 2.041 + 2.469 + 6.612 + 12.755 + 0.184 ) / 7
         = 41.714 / 7
         = 5.9592
  ```

  **The PC1 scores:**

  In 1D the only principal component is the data itself — the centred values are the PC1 scores. Explained variance ratio = 100% because there is nothing to reduce.

  ```
  PC1 scores: [−3.4286, −2.4286, −1.4286, +1.5714, +2.5714, +3.5714, −0.4286]
  ```

  **Visual Analogy:**

  Imagine all seven values on a number line. Mean centering slides the entire line so the average sits at zero. The spread of points around zero — their variance (5.9592) — is what PCA preserves when projecting. In 1D there is no compression to do; PCA simply standardizes the coordinate system.

  > **Note:** The real power of PCA emerges with two or more features, where it can find a lower-dimensional axis that captures most of the joint variance and allows projection without significant information loss.

</details>

---

## PCA (n features)

PCA with multiple features finds a new set of axes — the principal components — that are ordered by the amount of variance they capture. The first component points in the direction of greatest spread; the second points in the direction of greatest remaining spread, orthogonal to the first; and so on. Projecting onto the top k components reduces dimensionality from n to k.

**Projection onto k components:**

```
Z = X_centred @ W_k
```

Also written as:

```
C = (1/m) * X_centred^T * X_centred      (covariance matrix)
C * w = lambda * w                        (eigenvalue equation)
Z_i = x_centred_i · w_1, x_centred_i · w_2, ..., x_centred_i · w_k
```

Where `C` is the n×n covariance matrix, `w_j` is the j-th eigenvector (principal component direction), `lambda_j` is its eigenvalue (variance captured), and `Z_i` is the k-dimensional projection of point i — the new, compressed representation.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Compressing Student Data from 2 Features to 1</summary>
  <br/>

  Imagine reducing two features — hours studied (x1) and exam score (x2) — to a single principal component without losing important information.

  **Dataset:**

  | Student | Hours (x1) | Score (x2) |
  |---------|------------|------------|
  | S1      | 2          | 52         |
  | S2      | 3          | 55         |
  | S3      | 4          | 58         |
  | S4      | 7          | 74         |
  | S5      | 8          | 78         |
  | S6      | 9          | 82         |
  | S7      | 5          | 64         |

  **Step 1 — Mean-centre each feature:**

  ```
  mean(hours) = (2+3+4+7+8+9+5)/7 = 38/7 = 5.4286
  mean(score) = (52+55+58+74+78+82+64)/7 = 463/7 = 66.1429
  ```

  | Student | Centred hours       | Centred score         |
  |---------|---------------------|-----------------------|
  | S1      | 2 − 5.4286 = −3.4286| 52 − 66.1429 = −14.1429|
  | S2      | 3 − 5.4286 = −2.4286| 55 − 66.1429 = −11.1429|
  | S3      | 4 − 5.4286 = −1.4286| 58 − 66.1429 = −8.1429 |
  | S4      | 7 − 5.4286 = +1.5714| 74 − 66.1429 = +7.8571 |
  | S5      | 8 − 5.4286 = +2.5714| 78 − 66.1429 = +11.8571|
  | S6      | 9 − 5.4286 = +3.5714| 82 − 66.1429 = +15.8571|
  | S7      | 5 − 5.4286 = −0.4286| 64 − 66.1429 = −2.1429 |

  **Step 2 — Compute the covariance matrix:**

  ```
  Var(hours)       =  5.9592
  Var(score)       = 121.2653
  Cov(hours,score) =  26.7959

  C = [[ 5.9592,  26.7959],
       [26.7959, 121.2653]]
  ```

  The large covariance (26.7959) confirms that hours and score move together strongly — a student who studies more tends to score higher.

  **Step 3 — Compute eigenvalues and eigenvectors:**

  Solving `C * w = lambda * w`:

  ```
  Eigenvalue 1 (PC1): lambda_1 = 127.1882   → explains 99.97% of variance
  Eigenvalue 2 (PC2): lambda_2 =   0.0363   → explains  0.03% of variance

  PC1 direction: w1 = [0.2158,  0.9764]
  PC2 direction: w2 = [−0.9764, 0.2158]
  ```

  PC1 points mainly along the score axis (weight 0.9764) with a smaller hours component (0.2158) — it is the direction of maximum joint spread. PC2 is orthogonal to PC1 and captures almost nothing (0.03%).

  **Step 4 — Project each student onto PC1 and PC2:**

  ```
  PC1 score_i = centred_hours_i * 0.2158 + centred_score_i * 0.9764
  PC2 score_i = centred_hours_i * (−0.9764) + centred_score_i * 0.2158
  ```

  For S1 (centred = [−3.4286, −14.1429]):

  ```
  PC1 = (−3.4286)*0.2158 + (−14.1429)*0.9764
      = −0.7398 + (−13.8097)
      = −14.5495

  PC2 = (−3.4286)*(−0.9764) + (−14.1429)*0.2158
      = 3.3481 + (−3.0527)
      = 0.2954
  ```

  Full projection table:

  | Student | PC1 score | PC2 score |
  |---------|-----------|-----------|
  | S1      | −14.5495  | +0.2954   |
  | S2      | −11.4044  | −0.0336   |
  | S3      | −8.2593   | −0.3625   |
  | S4      | +8.0111   | +0.1614   |
  | S5      | +12.1327  | +0.0483   |
  | S6      | +16.2542  | −0.0649   |
  | S7      | −2.1849   | −0.0440   |

  **Step 5 — Keep only PC1 (dimensionality reduction from 2 → 1):**

  PC1 alone captures **99.97%** of the total variance. Dropping PC2 loses only 0.03% of information.

  **Step 6 — Reconstruct approximate original data from PC1:**

  ```
  X_approx = PC1_score * w1^T + mean
  ```

  For S1: `−14.5495 * [0.2158, 0.9764] + [5.4286, 66.1429]`

  ```
  = [−3.1402, −14.2066] + [5.4286, 66.1429]
  = [2.2884, 51.9363]
  ```

  Full reconstruction table:

  | Student | Original (hours, score) | Reconstructed (PC1 only) | Error           |
  |---------|-------------------------|--------------------------|-----------------|
  | S1      | (2.0000, 52.0000)       | (2.2884, 51.9363)        | (−0.2884, +0.0637) |
  | S2      | (3.0000, 55.0000)       | (2.9672, 55.0072)        | (+0.0328, −0.0072) |
  | S3      | (4.0000, 58.0000)       | (3.6460, 58.0782)        | (+0.3540, −0.0782) |
  | S4      | (7.0000, 74.0000)       | (7.1576, 73.9652)        | (−0.1576, +0.0348) |
  | S5      | (8.0000, 78.0000)       | (8.0471, 77.9896)        | (−0.0471, +0.0104) |
  | S6      | (9.0000, 82.0000)       | (8.9367, 82.0140)        | (+0.0633, −0.0140) |
  | S7      | (5.0000, 64.0000)       | (4.9570, 64.0095)        | (+0.0430, −0.0095) |

  The reconstruction errors are tiny — reducing from 2 features to 1 loses almost nothing in this dataset because hours and score are nearly perfectly correlated.

  ```mermaid
  flowchart LR
      A["Original space: 2 features - hours, score"]
      B["Mean centering - subtract feature means"]
      C["Covariance matrix C - 2x2"]
      D["Eigendecomposition - PC1 99.97 pct, PC2 0.03 pct"]
      E["Project onto PC1 - 1D scores"]
      F["Compressed space: 1 feature - PC1 score"]
      G["Reconstruct - approx original from PC1"]

      A --> B --> C --> D --> E --> F
      F --> G

      style A fill:#f0c040,stroke:#b8860b,color:#000
      style B fill:#f7b731,stroke:#e67e00,color:#000
      style C fill:#74c0fc,stroke:#1971c2,color:#000
      style D fill:#74c0fc,stroke:#1971c2,color:#000
      style E fill:#51cf66,stroke:#2f9e44,color:#000
      style F fill:#51cf66,stroke:#2f9e44,color:#000
      style G fill:#adb5bd,stroke:#6c757d,color:#000
  ```

  **Interpreting the result:**

  The PC1 score for each student is a single number that summarises both their hours studied and their exam score. Negative scores (S1–S3, S7) correspond to below-average students; positive scores (S4–S6) correspond to above-average students. The original two-dimensional scatter plot compresses into a single axis that captures 99.97% of the original variance — a near-lossless compression for this dataset.

  > **Note:** In datasets where features are less correlated, PC1 will explain less variance and more components will be needed. The threshold for how many components to keep is typically chosen to explain 95% or 99% of total variance.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

PCA finds only **linear** relationships between features — it cannot capture curved or non-linear structure in the data. If the meaningful variation is non-linear (e.g., a Swiss-roll manifold or concentric clusters), PCA will produce a poor low-dimensional representation. It is also sensitive to **feature scale** — a feature measured in thousands will dominate the covariance matrix and the first PC unless features are standardized beforehand. PCA produces components that are **linear combinations of all original features**, making them hard to interpret: PC1 being [0.22, 0.97] does not have an obvious real-world meaning. Finally, PCA is **unsupervised** — it maximizes variance without regard for class labels, so the most informative projection for reconstruction may not be the most informative projection for classification. In such cases, alternatives include t-SNE (non-linear, for 2D/3D visualization), UMAP (non-linear, faster than t-SNE, better at preserving global structure), Kernel PCA (non-linear PCA via kernel trick), Linear Discriminant Analysis (LDA — supervised, maximizes class separation rather than variance), or Autoencoders (neural network-based non-linear compression).

---

## Error and the Cost Function

### Reconstruction Error

PCA does not optimize a classification or regression loss. Its cost is the **reconstruction error** — how much information is lost by projecting onto k components and then projecting back.

| k (components kept) | Variance explained | RMSE     | Information lost |
|---------------------|--------------------|----------|-----------------|
| 1 (of 3 features)   | 99.82%             | 0.2819   | 0.18%           |
| 2 (of 3 features)   | 99.98%             | 0.1002   | 0.02%           |
| 3 (of 3 features)   | 100.00%            | 0.0000   | 0.00%           |

**Why minimize reconstruction error?**

- Keeping fewer components compresses the data and speeds up downstream models but introduces approximation error.
- The reconstruction error equals the sum of the eigenvalues of the **discarded** components: dropping small eigenvalues means dropping components that add little variance — and therefore little unique information.
- Squaring the error (MSE) ensures positive contributions from all dimensions and penalizes large deviations more than small ones.

---

### Objective: Maximize Retained Variance

The algorithm's goal is to find the k orthogonal directions that together capture the maximum possible proportion of the total variance in the data — equivalently, to minimize reconstruction error when projecting onto those k directions.

---

### Core Concept: Eigendecomposition of the Covariance Matrix

PCA builds its components through a single closed-form decomposition:

**Covariance matrix:**

$$C = \frac{1}{m} X_{\text{centred}}^T X_{\text{centred}}$$

**Eigenvalue equation:**

$$C \mathbf{w}_j = \lambda_j \mathbf{w}_j$$

**Explained variance ratio for component j:**

$$EVR_j = \frac{\lambda_j}{\sum_{i=1}^{n} \lambda_i}$$

**Example (2-feature covariance matrix):**

```
C = [[ 5.9592,  26.7959],
     [26.7959, 121.2653]]

Eigenvalue 1: lambda_1 = 127.1882   EVR_1 = 127.1882 / (127.1882 + 0.0363) = 99.97%
Eigenvalue 2: lambda_2 =   0.0363   EVR_2 = 0.0363   / (127.1882 + 0.0363) =  0.03%
```

The huge gap between `lambda_1` and `lambda_2` tells us that hours and score vary almost entirely along one direction — keeping only PC1 loses virtually nothing.

---

### Cost Function: Reconstruction MSE

The reconstruction error measures how well the k-component approximation reproduces the original data.

**Formula:**

$$MSE_{\text{recon}} = \frac{1}{m \cdot n} \sum_{i=1}^{m} \| x_i - \hat{x}_i \|^2$$

Where:
- `m` — number of data points
- `n` — number of original features
- `x_i` — original data point (n-dimensional)
- `x_hat_i` — reconstructed approximation using k components

**Example calculation (3-feature dataset, k=1):**

```
k=1: MSE = 0.0794  →  RMSE = sqrt(0.0794) = 0.2819
     Variance explained = 99.82%  →  Information lost = 0.18%

k=2: MSE = 0.0100  →  RMSE = sqrt(0.0100) = 0.1002
     Variance explained = 99.98%  →  Information lost = 0.02%
```

Adding PC2 reduces RMSE from 0.2819 to 0.1002 — a meaningful improvement for a 0.16% increase in explained variance.

---

### Alternative Notation — The Scree Plot

Since PCA produces as many components as there are features, a **Scree Plot** of eigenvalues (or explained variance ratios) against component number guides the choice of k. The "elbow" in the curve marks where adding more components yields diminishing returns.

**Eigenvalues and explained variance (3-feature dataset):**

| Component | Eigenvalue | Variance Explained | Cumulative |
|-----------|------------|-------------------|------------|
| PC1       | 130.2515   | 99.82%            | 99.82%     |
| PC2       | 0.2082     | 0.16%             | 99.98%     |
| PC3       | 0.0301     | 0.02%             | 100.00%    |

The elbow is at PC1: going from PC1 to PC2 adds only 0.16%, suggesting **k=1 is sufficient** for this dataset. A practical rule: choose the smallest k such that cumulative explained variance exceeds the chosen threshold (commonly 95% or 99%).

---

## How Do We Find the Principal Components?

PCA finds components through a **single closed-form computation** — no iterative optimization required. Given the mean-centred data matrix, the principal components are the eigenvectors of the covariance matrix, sorted by descending eigenvalue.

---

## PCA Algorithm

The algorithm runs in five deterministic steps.

**Pseudocode:**

```
Step 1 — Mean centering:
  mean_j = (1/m) * sum_i x_ij   for each feature j
  X_centred_ij = x_ij - mean_j

Step 2 — Covariance matrix:
  C = (1/m) * X_centred^T * X_centred   (n x n matrix)

Step 3 — Eigendecomposition:
  solve C * W = W * Lambda
  sort eigenvectors by eigenvalue descending:
    W = [w_1 | w_2 | ... | w_n]  (columns are eigenvectors)
    Lambda = diag(lambda_1, lambda_2, ..., lambda_n)

Step 4 — Select k components:
  choose k such that sum(lambda_1..lambda_k) / sum(all lambda) >= threshold
  W_k = W[:, :k]   (keep only top k eigenvectors)

Step 5 — Project:
  Z = X_centred @ W_k          (m x k projected data)

Step 6 — Reconstruct (optional):
  X_approx = Z @ W_k^T + mean  (m x n approximation)
```

**Example trace (2-feature dataset):**

```
Step 1: mean = [5.4286, 66.1429]
        S1 centred = [2−5.4286, 52−66.1429] = [−3.4286, −14.1429]

Step 2: C = [[5.9592, 26.7959],
             [26.7959, 121.2653]]

Step 3: lambda_1=127.1882 (w1=[0.2158, 0.9764])
        lambda_2=0.0363   (w2=[−0.9764, 0.2158])

Step 4: EVR_1 = 99.97% >= 95% threshold → keep k=1
        W_1 = [[0.2158], [0.9764]]

Step 5: S1 score = [−3.4286, −14.1429] @ [[0.2158], [0.9764]] = [−14.5495]

Step 6: S1 approx = [−14.5495] @ [[0.2158, 0.9764]] + [5.4286, 66.1429]
                   = [2.2884, 51.9363]
```

- A **large k** preserves more variance but reduces the compression benefit.
- A **small k** compresses more aggressively but increases reconstruction error.
- Features must be **standardized** (z-score) before PCA when they have very different scales — otherwise the feature with the largest absolute values dominates the covariance matrix and the first PC.

---

## Summary of Key Formulas

| Concept                      | Formula                                                                              |
|------------------------------|--------------------------------------------------------------------------------------|
| Mean centering               | x\_centred\_ij = x\_ij − mean\_j                                                    |
| Covariance matrix            | C = (1/m) * X\_centred^T * X\_centred                                               |
| Eigenvalue equation          | C * w\_j = lambda\_j * w\_j                                                          |
| Explained variance ratio     | EVR\_j = lambda\_j / sum(all lambda)                                                 |
| Projection (k components)    | Z = X\_centred @ W\_k                                                                |
| Reconstruction               | X\_approx = Z @ W\_k^T + mean                                                        |
| Reconstruction MSE           | MSE = (1/m*n) * sum\_i \|\| x\_i − x\_hat\_i \|\|^2                                 |
| Component selection rule     | choose smallest k s.t. sum(EVR\_1..k) >= threshold (e.g. 95%)                       |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- [PCA — Scikit-learn](https://github.com/gil-son/machine-learning/blob/main/unsupervised-learning/notebooks/dimension_reduction/pca/scikit-learn/PCA-v1.ipynb)
- [t-SNE and UMAP comparison](https://github.com/gil-son/machine-learning/blob/main/unsupervised-learning/notebooks/dimension_reduction/pca/scikit-learn/tSNE-UMAP-v1.ipynb)

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=FgakZw6K1QQ" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/FgakZw6K1QQ/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=g-Hb26agBFg" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/g-Hb26agBFg/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=fkf4IBRSeEc" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/fkf4IBRSeEc/hqdefault.jpg"/>
  </a>
</div>