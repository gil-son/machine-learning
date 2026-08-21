# K-Means Clustering <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

K-Means (1 feature) | K-Means (n features) | Choosing k — The Elbow Method

---

## What is K-Means? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

K-Means is an **unsupervised learning algorithm** used to discover **natural groupings** (clusters) in unlabeled data — for example, customer segments, document topics, or sensor anomalies.

At its core, it partitions the dataset into **k groups** by iteratively assigning each point to its nearest centroid and recomputing the centroids as the mean of their assigned points, until the assignments stop changing.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to find `k` centroids — one per cluster — such that every data point is assigned to its closest centroid and the total within-cluster variance is minimized. Three concepts are central:

- **Centroid** — the mean position of all points currently assigned to a cluster. It is recomputed after every assignment step and acts as the cluster's representative.
- **Assignment** — each point is assigned to the cluster whose centroid is nearest, measured by Euclidean distance.
- **WCSS (Within-Cluster Sum of Squares)** — the total squared distance from every point to its assigned centroid. K-Means minimizes this quantity.

**Distance formula (Euclidean):**

```
d(x, c) = sqrt( (x1-c1)^2 + (x2-c2)^2 + ... + (xn-cn)^2 )
```

**WCSS formula:**

```
WCSS = sum over all clusters k:
         sum over all points xi in cluster k:
           d(xi, centroid_k)^2
```

---

## K-Means (1 feature)

K-Means with a single feature partitions the number line into `k` intervals. At each iteration it measures the absolute distance from every point to every centroid, assigns each point to the nearest one, then recomputes centroids as the mean of their assigned points.

**Assignment rule:**

```
assign xi to cluster k* = argmin_k  |xi - c_k|
```

Also written as:

```
k*(xi) = argmin_k  d(xi, c_k)
c_k    = (1 / |C_k|) * sum of xi for all xi in C_k
```

Where:

- `xi` — the value of data point i
- `c_k` — the centroid of cluster k
- `k*` — the cluster with the nearest centroid to xi
- `|C_k|` — the number of points currently assigned to cluster k
- `d(xi, c_k)` — distance from point xi to centroid c_k

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Grouping Delivery Times into Fast and Slow Clusters</summary>
  <br/>

  Imagine grouping courier deliveries by time (in minutes) to automatically identify fast and slow deliveries — without knowing the labels in advance.

  **Dataset (Delivery Times in minutes):**

  | Point | Delivery Time (x) |
  |-------|-------------------|
  | P1    | 5                 |
  | P2    | 8                 |
  | P3    | 9                 |
  | P4    | 10                |
  | P5    | 30                |
  | P6    | 32                |
  | P7    | 35                |
  | P8    | 38                |

  **Step 1 — Initialize k=2 centroids:**

  Choose two initial centroids. A common strategy is to pick actual data points:

  ```
  C1 = 5.0   (first point — represents the "Fast" region)
  C2 = 30.0  (a distant point — represents the "Slow" region)
  ```

  **Step 2 — Assign each point to its nearest centroid (Iteration 1):**

  Compute the absolute distance from each point to C1 and C2:

  | Point | x  | \|x − C1\| | \|x − C2\| | Assigned to |
  |-------|----|-----------|-----------|-------------|
  | P1    | 5  | 0.0       | 25.0      | C1          |
  | P2    | 8  | 3.0       | 22.0      | C1          |
  | P3    | 9  | 4.0       | 21.0      | C1          |
  | P4    | 10 | 5.0       | 20.0      | C1          |
  | P5    | 30 | 25.0      | 0.0       | C2          |
  | P6    | 32 | 27.0      | 2.0       | C2          |
  | P7    | 35 | 30.0      | 5.0       | C2          |
  | P8    | 38 | 33.0      | 8.0       | C2          |

  **Step 3 — Recompute centroids:**

  ```
  C1_new = mean(5, 8, 9, 10) = 32 / 4 = 8.0
  C2_new = mean(30, 32, 35, 38) = 135 / 4 = 33.75
  ```

  **Step 4 — Assign again (Iteration 2) with updated centroids:**

  | Point | x  | \|x − C1=8.0\| | \|x − C2=33.75\| | Assigned to |
  |-------|----|--------------|----------------|-------------|
  | P1    | 5  | 3.0          | 28.75          | C1          |
  | P2    | 8  | 0.0          | 25.75          | C1          |
  | P3    | 9  | 1.0          | 24.75          | C1          |
  | P4    | 10 | 2.0          | 23.75          | C1          |
  | P5    | 30 | 22.0         | 3.75           | C2          |
  | P6    | 32 | 24.0         | 1.75           | C2          |
  | P7    | 35 | 27.0         | 1.25           | C2          |
  | P8    | 38 | 30.0         | 4.25           | C2          |

  Assignments are identical to Iteration 1 — **converged**.

  **The Fitted Clusters:**

  ```
  Cluster 1 (Fast deliveries): {5, 8, 9, 10}   centroid C1 = 8.00
  Cluster 2 (Slow deliveries): {30, 32, 35, 38} centroid C2 = 33.75
  ```

  **WCSS Calculation:**

  ```
  Cluster 1: (5−8)² + (8−8)² + (9−8)² + (10−8)² = 9 + 0 + 1 + 4 = 14.00
  Cluster 2: (30−33.75)² + (32−33.75)² + (35−33.75)² + (38−33.75)²
           = 14.0625 + 3.0625 + 1.5625 + 18.0625 = 36.75
  WCSS = 14.00 + 36.75 = 50.75
  ```

  **Prediction Example:**

  A new delivery takes 7 minutes. Which cluster does it belong to?

  ```
  d(7, C1=8.00)  = |7 − 8.00|  = 1.00
  d(7, C2=33.75) = |7 − 33.75| = 26.75
  → Assigned to Cluster 1 (Fast)
  ```

  **Visual Analogy:**

  Imagine all delivery times plotted on a number line. K-Means draws a dividing point between the two centroids (at `(8 + 33.75) / 2 = 20.875`) — every point to the left belongs to Cluster 1, every point to the right to Cluster 2. The centroids slide until they sit at the center of their respective groups.

  > **Note:** K-Means is sensitive to the initial centroid positions. Different initializations can lead to different final clusters. In practice, **K-Means++ initialization** is used to spread the initial centroids out, reducing the chance of poor convergence.

</details>

---

## K-Means (n features)

K-Means with multiple features works identically — the only change is that **distance is computed across all features simultaneously** using Euclidean distance in n-dimensional space. The centroid of each cluster is the mean position across all features.

**Assignment rule:**

```
assign xi to cluster k* = argmin_k  d(xi, c_k)

d(xi, c_k) = sqrt( sum_j (xi_j - c_k_j)^2 )
```

Also written as:

```
c_k = (1 / |C_k|) * sum of xi for all xi in C_k   (component-wise mean)
```

Each centroid is a vector with one component per feature, updated as the mean of all points in that cluster across every dimension simultaneously.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Grouping Students by Study Hours and Exam Score</summary>
  <br/>

  Imagine grouping students into clusters based on two features: hours studied (x1) and exam score (x2) — without using labels.

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
  | S8      | 6          | 68         |

  **Step 1 — Initialize k=2 centroids:**

  ```
  C1 = (2, 52)   (S1 — low hours, low score)
  C2 = (9, 82)   (S6 — high hours, high score)
  ```

  **Step 2 — Assign each point (Iteration 1):**

  Euclidean distance formula for 2 features:

  ```
  d(xi, c) = sqrt( (x1 - c1)^2 + (x2 - c2)^2 )
  ```

  Manual calculation for selected points:

  ```
  d(S1=(2,52), C1=(2,52)) = sqrt(0 + 0)    = 0.0000
  d(S1=(2,52), C2=(9,82)) = sqrt(49 + 900) = sqrt(949) = 30.8058

  d(S4=(7,74), C1=(2,52)) = sqrt(25 + 484) = sqrt(509) = 22.5610
  d(S4=(7,74), C2=(9,82)) = sqrt(4 + 64)   = sqrt(68)  =  8.2462

  d(S7=(5,64), C1=(2,52)) = sqrt(9 + 144)  = sqrt(153) = 12.3693
  d(S7=(5,64), C2=(9,82)) = sqrt(16 + 324) = sqrt(340) = 18.4391
  ```

  Full assignment table:

  | Student | d(C1)   | d(C2)   | Assigned |
  |---------|---------|---------|----------|
  | S1      | 0.0000  | 30.8058 | C1       |
  | S2      | 3.1623  | 27.6586 | C1       |
  | S3      | 6.3246  | 24.5153 | C1       |
  | S4      | 22.5610 | 8.2462  | C2       |
  | S5      | 26.6833 | 4.1231  | C2       |
  | S6      | 30.8058 | 0.0000  | C2       |
  | S7      | 12.3693 | 18.4391 | C1       |
  | S8      | 16.4924 | 14.3178 | C2       |

  **Step 3 — Recompute centroids:**

  ```
  C1_new = mean of {S1, S2, S3, S7}
         = ( (2+3+4+5)/4 ,  (52+55+58+64)/4 )
         = ( 14/4 , 229/4 )
         = ( 3.5 , 57.25 )

  C2_new = mean of {S4, S5, S6, S8}
         = ( (7+8+9+6)/4 ,  (74+78+82+68)/4 )
         = ( 30/4 , 302/4 )
         = ( 7.5 , 75.5 )
  ```

  **Step 4 — Assign again (Iteration 2) and check convergence:**

  | Student | d(C1=3.5,57.25) | d(C2=7.5,75.5) | Assigned |
  |---------|-----------------|----------------|----------|
  | S1      | 5.4601          | 24.1350        | C1       |
  | S2      | 2.3049          | 20.9881        | C1       |
  | S3      | 0.9014          | 17.8466        | C1       |
  | S4      | 17.1118         | 1.5811         | C2       |
  | S5      | 21.2323         | 2.5495         | C2       |
  | S6      | 25.3537         | 6.6708         | C2       |
  | S7      | 6.9147          | 11.7686        | C1       |
  | S8      | 11.0369         | 7.6485         | C2       |

  Assignments unchanged → **converged**.

  **The Fitted Clusters:**

  ```
  Cluster 1 (Low effort): S1, S2, S3, S7  →  centroid C1 = (3.5, 57.25)
  Cluster 2 (High effort): S4, S5, S6, S8 →  centroid C2 = (7.5, 75.5)
  ```

  **Prediction Example:**

  A new student studied 5.5 hours and scored 66. Which cluster?

  ```
  d((5.5, 66), C1=(3.5, 57.25)) = sqrt((5.5-3.5)^2 + (66-57.25)^2)
                                 = sqrt(4 + 76.5625)
                                 = sqrt(80.5625) = 8.9757

  d((5.5, 66), C2=(7.5, 75.5))  = sqrt((5.5-7.5)^2 + (66-75.5)^2)
                                 = sqrt(4 + 90.25)
                                 = sqrt(94.25) = 9.7082
  → Assigned to Cluster 1 (closer to C1)
  ```

  **Interpreting the clusters:**

  Each centroid describes the "typical" member of its cluster across all features simultaneously. C1 = (3.5, 57.25) represents a student who studied ~3.5 hours and scored ~57 — a lower-effort group. C2 = (7.5, 75.5) represents a student who studied ~7.5 hours and scored ~76 — a higher-effort group. K-Means discovered this structure without any labels.

  > **Note:** Features with very different scales (e.g., hours 1–10 vs income 0–100,000) dominate the distance calculation. Always **standardize features** (subtract mean, divide by std) before applying K-Means.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

K-Means requires specifying `k` in advance and assumes clusters are **roughly spherical and equal in size** — it struggles with elongated, irregular, or nested shapes. It is also sensitive to **outliers**, which can pull centroids away from the true cluster center, and to **initialization** — a poor starting point can converge to a suboptimal solution. WCSS is not convex in general, so K-Means finds a local minimum, not necessarily the global one. In such cases, alternatives include DBSCAN (density-based, finds arbitrary shapes and detects outliers), HDBSCAN (hierarchical density-based), Gaussian Mixture Models (soft cluster assignments with probabilistic membership), or Agglomerative Hierarchical Clustering (no need to specify k in advance).

---

## Error and the Cost Function

### Squared Distances

The **error** for each point is its squared distance to its assigned centroid. Squaring removes negatives, penalizes points far from their centroid more than nearby ones, and makes the total cost differentiable.

| Point | x  | Assigned Centroid | Distance | Squared Distance |
|-------|----|-------------------|----------|------------------|
| P1    | 5  | C1 = 8.00         | 3.00     | 9.00             |
| P2    | 8  | C1 = 8.00         | 0.00     | 0.00             |
| P5    | 30 | C2 = 33.75        | 3.75     | 14.06            |
| P8    | 38 | C2 = 33.75        | 4.25     | 18.06            |

**Why square the distances?**

- **To prevent cancellation:** A point 3 units above and a point 3 units below the centroid would sum to zero — squaring ensures both contribute positively.
- **To penalize outliers more:** A point 6 units away contributes 36; a point 3 units away contributes only 9. This encourages compact, tight clusters.

---

### Objective: Minimize the WCSS

The algorithm's goal is to find centroid positions and cluster assignments that make the total within-cluster variance as small as possible across all k clusters.

---

### WCSS — Within-Cluster Sum of Squares

- **Definition:** The sum of squared distances from every point to its assigned centroid, totalled across all clusters.
- **Formula:**

$$WCSS = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

- **Interpretation:** Measures total compactness of all clusters. Lower WCSS = tighter, more separated clusters.

Using the 1-feature example (k=2):

$$WCSS = \underbrace{(5-8)^2 + (8-8)^2 + (9-8)^2 + (10-8)^2}_{\text{Cluster 1} = 14.00} + \underbrace{(30-33.75)^2 + (32-33.75)^2 + (35-33.75)^2 + (38-33.75)^2}_{\text{Cluster 2} = 36.75} = 50.75$$

---

### Cost Function: Mean Squared Distance (Inertia)

The WCSS is also called **inertia** in scikit-learn. Normalizing by the number of points gives the mean squared distance, which is comparable across datasets of different sizes.

**Formula:**

$$J(\mu_1, ..., \mu_K) = \frac{1}{m} \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

Where:
- `m` — total number of data points
- `mu_k` — centroid of cluster k
- `K` — number of clusters

**Example Calculation (1-feature, k=2, m=8):**

$$J = \frac{WCSS}{m} = \frac{50.75}{8} = 6.3438$$

To express in original units:

$$\sqrt{J} = \sqrt{6.3438} \approx 2.519 \text{ minutes}$$

On average, each point is about **2.52 minutes** from its cluster centroid.

---

### Alternative Notation — The Elbow Method

WCSS always decreases as k increases — with k = m (one cluster per point), WCSS = 0. The Elbow Method plots WCSS against k and looks for the **"elbow"**: the point where adding another cluster produces diminishing returns.

**WCSS for the 2-feature student dataset:**

| k | WCSS    | Drop from previous |
|---|---------|-------------------|
| 1 | 893.88  | —                 |
| 2 | 195.75  | −698.13           |
| 3 | 62.50   | −133.25           |
| 4 | 37.00   | −25.50            |
| 5 | 22.00   | −15.00            |

The largest drop is from k=1 to k=2 (−698.13), and the next biggest is k=2 to k=3 (−133.25). The curve flattens after k=2, suggesting **k=2 is the natural number of clusters** in this dataset. Gradient descent on the WCSS cost function adjusts `mu_k` to reduce this value — the optimal centroid for a fixed assignment is always the mean of the assigned points.

---

## How Do We Find the Best Centroids?

The WCSS cost function is minimized using **Lloyd's Algorithm** — an iterative two-step method that alternates between assigning points to their nearest centroid and recomputing centroids as the mean of their current members.

---

## Lloyd's Algorithm (K-Means Update)

Lloyd's Algorithm updates centroid positions at each iteration by computing the mean of all currently assigned points, then reassigning all points to the nearest updated centroid.

**Update Rule:**

$$\mu_k := \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

This is applied simultaneously to all k centroids after each full assignment pass.

**Example:**

> Note: The initial centroids C1 = (2, 52) and C2 = (9, 82) below are chosen from the data. In practice, **K-Means++ initialization** picks centroids that are spread far apart, reducing the risk of poor convergence.

Continuing from the 2-feature example, after Iteration 1 the assignments are:

- C1 members: S1=(2,52), S2=(3,55), S3=(4,58), S7=(5,64)
- C2 members: S4=(7,74), S5=(8,78), S6=(9,82), S8=(6,68)

**Update C1:**

```
mu1_x1 = (2 + 3 + 4 + 5) / 4 = 14 / 4 = 3.5
mu1_x2 = (52 + 55 + 58 + 64) / 4 = 229 / 4 = 57.25
C1_new = (3.5, 57.25)
```

**Update C2:**

```
mu2_x1 = (7 + 8 + 9 + 6) / 4 = 30 / 4 = 7.5
mu2_x2 = (74 + 78 + 82 + 68) / 4 = 302 / 4 = 75.5
C2_new = (7.5, 75.5)
```

After the update, all points are reassigned to the nearest new centroid. Since the assignments did not change between Iteration 1 and Iteration 2, the algorithm has converged. Final centroids: C1 = (3.5, 57.25), C2 = (7.5, 75.5).

- A **large k** leads to lower WCSS but more centroids than may be meaningful.
- A **small k** leads to higher WCSS, merging distinct groups into one.
- **K-Means++ initialization** reduces sensitivity to starting positions by placing initial centroids far from each other with probability proportional to distance squared.

---

## Summary of Key Formulas

| Concept                     | Formula                                                                          |
|-----------------------------|----------------------------------------------------------------------------------|
| Distance (Euclidean)        | d(x, c) = sqrt( sum_j (x_j - c_j)^2 )                                           |
| Assignment rule             | k*(xi) = argmin_k  d(xi, mu_k)                                                   |
| Centroid update             | mu_k = (1 / \|C_k\|) * sum of xi for all xi in C_k                              |
| WCSS (cost function)        | J = sum_k sum_{xi in C_k} \|\|xi - mu_k\|\|^2                                   |
| Mean squared distance       | J_mean = WCSS / m                                                                |
| Convergence condition       | assignments unchanged between two consecutive iterations                         |
| Elbow method                | plot WCSS vs k; choose k at the point of diminishing returns                     |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- K-Means Clustering — Scikit-learn *(coming soon)*

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=4b5d3muPQmA" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/4b5d3muPQmA/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=R2e3Ls9H_fc" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/R2e3Ls9H_fc/hqdefault.jpg"/>
  </a>
</div>