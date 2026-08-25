# DBSCAN <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

DBSCAN (1 feature) | DBSCAN (n features) | eps and min_pts | Noise Detection

---

## What is DBSCAN? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an **unsupervised learning algorithm** that discovers clusters of **arbitrary shape** in unlabeled data and explicitly identifies **noise points** that do not belong to any cluster.

At its core, DBSCAN does not require specifying the number of clusters in advance. Instead, it grows clusters by density: a region is a cluster if it contains enough points within a given radius. Points in sparse regions are labeled as noise. This makes it especially powerful for datasets where clusters are irregularly shaped, of varying size, or embedded in a noisy background.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to find all dense regions in the data and group them into clusters, while flagging sparse points as noise. Three concepts define every point's role:

- **Core point** — a point with at least `min_pts` neighbours within distance `eps` (including itself). Core points are the seeds from which clusters grow.
- **Border point** — a point with fewer than `min_pts` neighbours within `eps`, but within `eps` of at least one core point. It belongs to a cluster but cannot expand it.
- **Noise point** — a point that is neither a core point nor a border point. It belongs to no cluster.

**Two hyperparameters control everything:**

```
eps      — the neighbourhood radius: how close two points must be to be considered neighbours
min_pts  — the minimum number of neighbours a point needs within eps to be a core point
```

**Neighbourhood formula:**

```
N_eps(xi) = { xj | d(xi, xj) <= eps }
```

**Distance formula (Euclidean):**

```
d(xi, xj) = sqrt( (xi1-xj1)^2 + (xi2-xj2)^2 + ... + (xin-xjn)^2 )
```

---

## DBSCAN (1 feature)

DBSCAN with a single feature scans the number line for dense stretches of points. A point is a core point if at least `min_pts` other points lie within distance `eps` of it. Connected core points and their border points form a cluster; isolated points with no dense neighbourhood become noise.

**Core point rule:**

```
xi is a core point  if  |N_eps(xi)| >= min_pts
```

Also written as:

```
N_eps(xi) = { xj in X | |xi - xj| <= eps,  j != i }

xi is Core   if |N_eps(xi)| >= min_pts
xi is Border if |N_eps(xi)| <  min_pts  AND  xi in N_eps(xc) for some core xc
xi is Noise  if |N_eps(xi)| <  min_pts  AND  xi not reachable from any core point
```

Where:

- `xi` — the point being evaluated
- `eps` — neighbourhood radius
- `min_pts` — minimum neighbours to be a core point
- `N_eps(xi)` — the set of neighbours of xi within distance eps
- `|N_eps(xi)|` — the count of those neighbours

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Grouping Delivery Times with Noise Detection</summary>
  <br/>

  Imagine grouping courier deliveries by time (in minutes) — including one outlier delivery that doesn't fit either group.

  **Dataset (Delivery Times in minutes):**

  | Point | Time (x) |
  |-------|----------|
  | P1    | 5        |
  | P2    | 8        |
  | P3    | 9        |
  | P4    | 10       |
  | P5    | 30       |
  | P6    | 32       |
  | P7    | 35       |
  | P8    | 38       |
  | P9    | 22       |

  **Parameters:** `eps = 5.0`, `min_pts = 2`

  **Step 1 — Compute neighbourhood counts:**

  For each point, count how many other points lie within distance 5.0:

  | Point | x  | Neighbours within eps=5 | Count | Role   |
  |-------|----|--------------------------|-------|--------|
  | P1    | 5  | {8, 9, 10}               | 3     | Core   |
  | P2    | 8  | {5, 9, 10}               | 3     | Core   |
  | P3    | 9  | {5, 8, 10}               | 3     | Core   |
  | P4    | 10 | {5, 8, 9}                | 3     | Core   |
  | P5    | 30 | {32, 35}                 | 2     | Core   |
  | P6    | 32 | {30, 35}                 | 2     | Core   |
  | P7    | 35 | {30, 32, 38}             | 3     | Core   |
  | P8    | 38 | {35}                     | 1     | Border |
  | P9    | 22 | {}                       | 0     | Noise  |

  **Step 2 — Grow Cluster 1 from P1:**

  P1 is a core point. Start Cluster 1 and add all points density-reachable from P1:

  ```
  Start: Cluster 1 = {P1}
  P1 neighbours: {P2, P3, P4}  → all are core points → add to Cluster 1
  P2 neighbours: {P1, P3, P4}  → already in Cluster 1 → no expansion
  P3 neighbours: {P1, P2, P4}  → already in Cluster 1 → no expansion
  P4 neighbours: {P1, P2, P3}  → already in Cluster 1 → no expansion
  Result: Cluster 1 = {P1, P2, P3, P4}
  ```

  **Step 3 — Grow Cluster 2 from P5:**

  P5 is unvisited and a core point. Start Cluster 2:

  ```
  Start: Cluster 2 = {P5}
  P5 neighbours: {P6, P7}  → both core → add to Cluster 2
  P6 neighbours: {P5, P7}  → already in Cluster 2
  P7 neighbours: {P5, P6, P8}  → P8 has 1 neighbour < min_pts → Border → add to Cluster 2
  Result: Cluster 2 = {P5, P6, P7, P8}
  ```

  **Step 4 — Mark P9 as Noise:**

  ```
  P9 (x=22): 0 neighbours within eps=5  →  NOISE
  ```

  **The Final Clusters:**

  ```
  Cluster 1 (Fast deliveries):  {P1=5, P2=8, P3=9, P4=10}
  Cluster 2 (Slow deliveries):  {P5=30, P6=32, P7=35, P8=38}
  Noise:                         {P9=22}
  ```

  **Prediction Example:**

  A new delivery takes 7 minutes. Which cluster?

  ```
  d(7, P1=5)  = 2.0  <= eps=5  (Cluster 1 core point)
  d(7, P2=8)  = 1.0  <= eps=5  (Cluster 1 core point)
  d(7, P9=22) = 15.0 > eps=5   (noise, irrelevant)

  → New point is within eps of a Cluster 1 core point → assigned to Cluster 1
  ```

  **Visual Analogy:**

  Imagine all delivery times plotted on a number line. DBSCAN draws a circle of radius 5 around each point and asks: "Does this circle contain at least 2 other points?" Dense stretches (5–10 and 30–38) become clusters; the isolated point at 22 — too far from both groups — becomes noise. No pre-specified number of clusters was needed.

  > **Note:** Unlike K-Means, DBSCAN does not assign noise points to the nearest cluster. P9 at x=22 is simply labeled noise because it falls in a sparse region. This is a feature, not a limitation — it prevents outliers from polluting cluster statistics.

</details>

---

## DBSCAN (n features)

DBSCAN with multiple features works identically — the only change is that **Euclidean distance is computed across all features simultaneously**. The neighbourhood of a point becomes a hypersphere of radius `eps` in n-dimensional space, and the same core / border / noise classification applies.

**Core point rule (n features):**

```
xi is a core point  if  |N_eps(xi)| >= min_pts

N_eps(xi) = { xj | sqrt( sum_f (xi_f - xj_f)^2 ) <= eps,  j != i }
```

Also written as:

```
d(xi, xj) = || xi - xj ||₂   (L2 norm across all features)
```

Because DBSCAN uses only pairwise distances, it finds clusters of any shape — not just spheres — as long as the points within a cluster are connected through a chain of core-to-core density reachability.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Grouping Students by Study Hours and Exam Score, with Noise</summary>
  <br/>

  Imagine grouping students based on two features — hours studied (x1) and exam score (x2) — where one student is an outlier who studied few hours but scored unusually low.

  **Dataset:**

  | Student | Hours (x1) | Score (x2) |
  |---------|------------|------------|
  | A1      | 2          | 52         |
  | A2      | 3          | 55         |
  | A3      | 4          | 58         |
  | B1      | 7          | 74         |
  | B2      | 8          | 78         |
  | B3      | 9          | 82         |
  | M1      | 5          | 64         |
  | N1      | 15         | 40         |

  **Parameters:** `eps = 8.0`, `min_pts = 2`

  **Step 1 — Compute all pairwise distances:**

  Manual calculation for key pairs:

  ```
  d(A1, A2) = sqrt((2-3)^2  + (52-55)^2)  = sqrt(1  + 9)   = sqrt(10)  = 3.1623
  d(A1, A3) = sqrt((2-4)^2  + (52-58)^2)  = sqrt(4  + 36)  = sqrt(40)  = 6.3246
  d(A3, M1) = sqrt((4-5)^2  + (58-64)^2)  = sqrt(1  + 36)  = sqrt(37)  = 6.0828
  d(B1, B2) = sqrt((7-8)^2  + (74-78)^2)  = sqrt(1  + 16)  = sqrt(17)  = 4.1231
  d(B2, B3) = sqrt((8-9)^2  + (78-82)^2)  = sqrt(1  + 16)  = sqrt(17)  = 4.1231
  d(B1, B3) = sqrt((7-9)^2  + (74-82)^2)  = sqrt(4  + 64)  = sqrt(68)  = 8.2462
  d(A1, N1) = sqrt((2-15)^2 + (52-40)^2)  = sqrt(169+144)  = sqrt(313) = 17.6918
  d(M1, B1) = sqrt((5-7)^2  + (64-74)^2)  = sqrt(4  + 100) = sqrt(104) = 10.1980
  ```

  **Step 2 — Classify each point:**

  Count neighbours within eps = 8.0:

  | Student | Neighbours within eps=8        | Count | Role   |
  |---------|--------------------------------|-------|--------|
  | A1      | A2 (3.16), A3 (6.32)           | 2     | Core   |
  | A2      | A1 (3.16), A3 (3.16)           | 2     | Core   |
  | A3      | A1 (6.32), A2 (3.16), M1 (6.08)| 3     | Core   |
  | B1      | B2 (4.12)                      | 1     | Border |
  | B2      | B1 (4.12), B3 (4.12)           | 2     | Core   |
  | B3      | B2 (4.12)                      | 1     | Border |
  | M1      | A3 (6.08)                      | 1     | Border |
  | N1      | (none)                         | 0     | Noise  |

  **Step 3 — Grow Cluster 1 from A1:**

  ```
  A1 is core → start Cluster 1
  A1 neighbours: {A2, A3}  → both core → add and expand
    A2 neighbours: {A1, A3} → already in Cluster 1
    A3 neighbours: {A1, A2, M1} → M1 has 1 neighbour (< min_pts) → Border → add to Cluster 1
  Cluster 1 = {A1, A2, A3, M1}
  ```

  **Step 4 — Grow Cluster 2 from B2:**

  ```
  B2 is core → start Cluster 2
  B2 neighbours: {B1, B3}
    B1: 1 neighbour → Border → add to Cluster 2
    B3: 1 neighbour → Border → add to Cluster 2
  Cluster 2 = {B1, B2, B3}
  ```

  **Step 5 — Mark N1 as Noise:**

  ```
  N1 (15, 40): 0 neighbours within eps=8  →  NOISE
  ```

  **The Final Clusters:**

  ```mermaid
  flowchart TD
      ROOT["DBSCAN - eps=8.0, min_pts=2"]
      C1["Cluster 1 - Low effort group | A1=2,52 | A2=3,55 | A3=4,58 | M1=5,64"]
      C2["Cluster 2 - High effort group | B1=7,74 | B2=8,78 | B3=9,82"]
      NOISE["NOISE | N1=15,40 - isolated outlier"]

      ROOT --> C1
      ROOT --> C2
      ROOT --> NOISE

      style C1 fill:#74c0fc,stroke:#1971c2,color:#000
      style C2 fill:#51cf66,stroke:#2f9e44,color:#000
      style NOISE fill:#ff6b6b,stroke:#c0392b,color:#fff
      style ROOT fill:#f0c040,stroke:#b8860b,color:#000
  ```

  **Step 6 — Prediction Example:**

  A new student studied 3.5 hours and scored 57. Which cluster?

  ```
  d(new=(3.5,57), A1=(2,52))  = sqrt(2.25 + 25)  = sqrt(27.25) = 5.2202
  d(new=(3.5,57), A2=(3,55))  = sqrt(0.25 + 4)   = sqrt(4.25)  = 2.0616
  d(new=(3.5,57), A3=(4,58))  = sqrt(0.25 + 1)   = sqrt(1.25)  = 1.1180
  d(new=(3.5,57), M1=(5,64))  = sqrt(2.25 + 49)  = sqrt(51.25) = 7.1589
  ```

  A1, A2, A3 and M1 are all within eps=8.0, and A3 is a core point of Cluster 1.

  ```
  → New point is density-reachable from Cluster 1 core points → Cluster 1
  ```

  **Interpreting the result:**

  DBSCAN found two natural groups without being told k=2. It also flagged N1 as noise — a student whose combination of hours (15) and score (40) is completely isolated from both groups. A K-Means model would have forced N1 into the nearest cluster, silently distorting it.

  > **Note:** DBSCAN assigns a new point to a cluster only if it falls within `eps` of at least one core point of that cluster. If the new point is in a sparse region — within eps of a border point only, or of no cluster point at all — it is labeled noise.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

DBSCAN requires choosing `eps` and `min_pts` — and results are highly sensitive to both. A single global `eps` struggles when clusters have **varying density**: a tight cluster and a sparse cluster may need very different radii to be correctly identified. DBSCAN also scales poorly to **high-dimensional data** because the concept of a meaningful neighbourhood radius breaks down as dimensionality increases (the curse of dimensionality). It is not designed for **incremental updates** — adding new data requires re-running the algorithm from scratch. In such cases, alternatives include HDBSCAN (which adapts density thresholds per cluster and is more robust to varying density), OPTICS (which produces a reachability plot rather than a hard radius), Gaussian Mixture Models (for soft probabilistic cluster membership), or K-Means (when clusters are roughly spherical, equal-sized, and the number of clusters is known).

---

## Error and the Cost Function

### Why DBSCAN Has No Traditional Cost Function

Unlike K-Means (which minimizes WCSS) or logistic regression (which minimizes log loss), DBSCAN does not optimize a differentiable cost function. It is a **deterministic rule-based algorithm**: given fixed `eps` and `min_pts`, the output is always the same regardless of initialization. There are no weights to update, no gradients to compute.

Instead, cluster quality is measured **after the fact** using external validation metrics:

| Metric                    | Definition                                                              | Range       |
|---------------------------|-------------------------------------------------------------------------|-------------|
| Silhouette Score          | Mean ratio of intra-cluster cohesion to inter-cluster separation        | −1 to +1    |
| Noise ratio               | Fraction of points labeled as noise                                     | 0 to 1      |
| Number of clusters found  | How many clusters DBSCAN discovered                                     | ≥ 0         |
| Davies-Bouldin Index      | Average similarity of each cluster to its most similar cluster (lower = better) | ≥ 0   |

---

### Objective: Find All Dense Regions

The algorithm's goal is to identify every connected dense region in the data — where "dense" means a neighbourhood of radius `eps` contains at least `min_pts` points — and label all remaining sparse points as noise.

---

### Core Concept: Density Reachability

DBSCAN builds clusters through a chain of **density reachability**:

- **Direct density reachability:** xj is directly reachable from xi if xj is in N_eps(xi) and xi is a core point.
- **Density reachability:** xj is reachable from xi if there is a chain xi → p1 → p2 → ... → xj where each step is direct density reachability.
- **Density connectivity:** xi and xj are in the same cluster if there exists a point xo from which both xi and xj are density reachable.

**Formula — point classification:**

$$\text{Core: } |N_\varepsilon(x_i)| \geq \text{min\_pts}$$

$$\text{Border: } |N_\varepsilon(x_i)| < \text{min\_pts} \text{ AND } \exists \text{ core } x_c : d(x_i, x_c) \leq \varepsilon$$

$$\text{Noise: } |N_\varepsilon(x_i)| < \text{min\_pts} \text{ AND } \nexists \text{ core } x_c : d(x_i, x_c) \leq \varepsilon$$

**Example (2-feature dataset, eps=8, min\_pts=2):**

```
A3 = (4, 58)
N_eps(A3) = {A1 (d=6.32), A2 (d=3.16), M1 (d=6.08)}
|N_eps(A3)| = 3 >= 2  →  Core point
```

```
M1 = (5, 64)
N_eps(M1) = {A3 (d=6.08)}
|N_eps(M1)| = 1 < 2  →  not Core
But M1 is within eps of A3, which is Core  →  Border point
```

```
N1 = (15, 40)
N_eps(N1) = {}
|N_eps(N1)| = 0 < 2  →  Noise
Not within eps of any core point  →  Noise
```

---

### Effect of eps and min\_pts

The two parameters jointly determine what counts as "dense" and therefore what gets called a cluster:

**Effect of eps (with min\_pts = 2):**

| eps   | Clusters found | Noise points | Behaviour                                            |
|-------|---------------|--------------|------------------------------------------------------|
| 3.0   | 0             | 8            | eps too small — no point has enough neighbours       |
| 6.0   | 2             | 2            | Finds A-group and B-group; M1 and N1 are noise       |
| **8.0** | **2**       | **1**        | **Finds both groups; M1 joins Cluster 1; N1 = noise**|
| 12.0  | 1             | 1            | eps too large — A and B groups merge into one        |

**Effect of min\_pts (with eps = 8.0):**

| min\_pts | Clusters found | Noise points | Behaviour                                           |
|----------|---------------|--------------|-----------------------------------------------------|
| 2        | 2             | 1            | Both groups found; border points included           |
| 3        | 1             | 4            | Only A3 qualifies as core (3 neighbours); A1, A2, B1, B3 become noise |

A useful rule of thumb: set `min_pts = 2 * n_features` as a starting point, then tune `eps` using a **k-distance plot** — plot the distance to the k-th nearest neighbour for each point, sorted in ascending order. The elbow of this curve is a good estimate for `eps`.

---

### Alternative Notation (Silhouette Score)

Since DBSCAN has no training loss, the **Silhouette Score** is the standard way to quantify cluster quality after fitting:

$$s(x_i) = \frac{b(x_i) - a(x_i)}{\max(a(x_i),\ b(x_i))}$$

Where:
- `a(xi)` — mean distance from xi to all other points **in the same cluster** (cohesion)
- `b(xi)` — mean distance from xi to all points **in the nearest other cluster** (separation)
- `s(xi)` ranges from −1 (wrong cluster) to +1 (well-separated)

**Example (point A2=(3,55), Cluster 1 = {A1,A2,A3,M1}, Cluster 2 = {B1,B2,B3}):**

```
a(A2) = mean( d(A2,A1), d(A2,A3), d(A2,M1) )
       = mean( 3.1623, 3.1623, 9.2195 )
       = 15.5441 / 3 = 5.1814

b(A2) = mean( d(A2,B1), d(A2,B2), d(A2,B3) )
       = mean( 19.4165, 23.5372, 27.6586 )
       = 70.6123 / 3 = 23.5374

s(A2) = (23.5374 - 5.1814) / max(5.1814, 23.5374)
       = 18.3560 / 23.5374
       = 0.7799
```

A silhouette score of **0.78** for A2 means it is well-placed in its cluster — much closer to its own cluster members than to Cluster 2. Scores near 1 across all points indicate well-separated, compact clusters.

---

## How Do We Find the Clusters?

DBSCAN finds clusters using a **single linear scan** of the data combined with neighbourhood queries. There are no iterations in the K-Means sense — once `eps` and `min_pts` are set, the algorithm makes one pass through the data, expanding each unvisited core point into a cluster before moving to the next.

---

## DBSCAN Algorithm

The algorithm visits each point exactly once, classifies it, and either expands a new cluster or marks it as noise.

**Pseudocode:**

```
for each unvisited point xi:
    mark xi as visited
    N = neighbourhood_query(xi, eps)

    if |N| < min_pts:
        label xi as NOISE

    else:
        cluster_id += 1
        label xi as cluster_id          ← xi is a core point
        expand_cluster(xi, N, cluster_id)

expand_cluster(xi, N, cluster_id):
    for each xj in N:
        if xj not visited:
            mark xj as visited
            N_j = neighbourhood_query(xj, eps)
            if |N_j| >= min_pts:         ← xj is also a core point
                N = N + N_j              ← grow the neighbourhood queue
        if xj not yet assigned to any cluster:
            label xj as cluster_id
```

**Example (2-feature, expanding Cluster 1 from A1):**

```
Visit A1=(2,52): N = {A2, A3}  |N|=2 >= 2  → Core → Cluster 1
  Visit A2=(3,55): N_A2 = {A1, A3}  |N_A2|=2 >= 2 → Core → add {A1,A3} to queue (already in)
  Visit A3=(4,58): N_A3 = {A1, A2, M1}  |N_A3|=3 >= 2 → Core → add M1 to queue
    Visit M1=(5,64): N_M1 = {A3}  |N_M1|=1 < 2 → Border → label as Cluster 1, do not expand
Queue exhausted → Cluster 1 = {A1, A2, A3, M1}
```

- A **small eps** creates many small clusters and marks most points as noise.
- A **large eps** merges distinct clusters into one.
- A **small min\_pts** makes it easier to be a core point — more clusters, fewer noise points.
- A **large min\_pts** requires denser regions — fewer, stricter clusters, more noise.

---

## Summary of Key Formulas

| Concept                    | Formula                                                                              |
|----------------------------|--------------------------------------------------------------------------------------|
| Neighbourhood              | N\_eps(xi) = { xj \| d(xi, xj) <= eps, j != i }                                     |
| Euclidean distance         | d(xi, xj) = sqrt( sum\_f (xi\_f - xj\_f)^2 )                                        |
| Core point condition       | \|N\_eps(xi)\| >= min\_pts                                                            |
| Border point condition     | \|N\_eps(xi)\| < min\_pts AND within eps of a core point                              |
| Noise point condition      | \|N\_eps(xi)\| < min\_pts AND NOT within eps of any core point                        |
| Silhouette score (point)   | s(xi) = (b(xi) - a(xi)) / max(a(xi), b(xi))                                         |
| Silhouette score (dataset) | S = (1/m) * sum\_i s(xi)  (excluding noise points)                                   |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- DBSCAN — Scikit-learn - *(coming soon)*
---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=RDZUdRSDOok" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/RDZUdRSDOok/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=C3r7tGRe2eI" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/C3r7tGRe2eI/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=dGsxd67IFiU" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/dGsxd67IFiU/hqdefault.jpg"/>
  </a>
</div>