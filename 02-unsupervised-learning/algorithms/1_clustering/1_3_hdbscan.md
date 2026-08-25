# HDBSCAN <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

HDBSCAN (1 feature) | HDBSCAN (n features) | Hierarchy and Stability | Soft Clustering

---

## What is HDBSCAN? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is an **unsupervised learning algorithm** that extends DBSCAN by building a full **hierarchy of density-based clusters** and then extracting the most stable ones — automatically and without requiring a fixed `eps` radius.

At its core, HDBSCAN transforms the data into a density landscape, connects points through a **Minimum Spanning Tree** of mutual reachability distances, builds a **condensed cluster hierarchy**, and selects clusters based on their **stability** — how long they persist as the density threshold varies. This eliminates the most fragile limitation of DBSCAN: the need to choose a single global `eps` that works for all clusters simultaneously.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to extract the most persistent, stable clusters from data that may contain regions of varying density, while flagging sparse points as noise. Four concepts are central:

- **Core distance** — the distance from a point to its `min_cluster_size`-th nearest neighbour. Points in dense regions have small core distances; isolated points have large ones.
- **Mutual reachability distance (MRD)** — a smoothed distance between two points that respects the local density around each: `mrd(xi, xj) = max(core_dist(xi), core_dist(xj), d(xi, xj))`. It prevents noisy isolated points from accidentally connecting to clusters.
- **Minimum Spanning Tree (MST)** — a tree connecting all points with the minimum total MRD. Cutting this tree from the largest edges downward builds the cluster hierarchy.
- **Cluster stability** — the sum of how long each point survives within its cluster as the density threshold increases. Clusters with higher stability are selected as the final output.

**Only one required hyperparameter:**

```
min_cluster_size  — the minimum number of points for a group to be considered a cluster
```

**Core distance formula:**

```
core_dist_k(xi) = distance to the k-th nearest neighbour of xi
                  where k = min_cluster_size
```

**Mutual reachability distance formula:**

```
mrd(xi, xj) = max( core_dist(xi),  core_dist(xj),  d(xi, xj) )
```

---

## HDBSCAN (1 feature)

HDBSCAN with a single feature scans the number line for density structure at multiple scales. It computes core distances, builds mutual reachability distances between all pairs, extracts the MST, and then cuts the MST from the largest edges inward to build the cluster hierarchy — automatically selecting the most stable cuts.

**Core distance (1D):**

```
core_dist(xi) = distance to the min_cluster_size-th nearest neighbour of xi
```

Also written as:

```
mrd(xi, xj) = max( core_dist(xi),  core_dist(xj),  |xi - xj| )

lambda(xi, xj) = 1 / mrd(xi, xj)   (density level at which xi and xj connect)
```

Where:

- `xi` — value of data point i
- `core_dist(xi)` — density radius around xi; small in dense areas, large in sparse areas
- `mrd(xi, xj)` — effective distance between xi and xj after density smoothing
- `lambda` — the inverse of MRD; higher lambda = points connect at higher density = tighter cluster

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Grouping Delivery Times with Varying Density</summary>
  <br/>

  Imagine grouping courier deliveries by time (in minutes). Unlike DBSCAN, HDBSCAN handles the fact that the Fast cluster is tighter (5–10 min) while the Slow cluster is more spread out (30–38 min) — different local densities.

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

  **Parameter:** `min_cluster_size = 2`

  **Step 1 — Compute core distances:**

  For each point, find the distance to its 2nd nearest neighbour:

  | Point | x  | 2nd-NN distance | core\_dist |
  |-------|----|-----------------|------------|
  | P1    | 5  | 4 (to P4=10 − no, to P2=8: gap 3, P3=9: gap 4) | 4.00 |
  | P2    | 8  | 2 (to P3=9 or P1=5, closer ones) | 2.00 |
  | P3    | 9  | 1 (P2=8 and P4=10 both at dist 1; 2nd-NN = 1) | 1.00 |
  | P4    | 10 | 2 (P3=9 at 1, P2=8 at 2) | 2.00 |
  | P5    | 30 | 5 (P6=32 at 2, P7=35 at 5) | 5.00 |
  | P6    | 32 | 3 (P5=30 at 2, P7=35 at 3) | 3.00 |
  | P7    | 35 | 3 (P6=32 at 3, P8=38 at 3) | 3.00 |
  | P8    | 38 | 6 (P7=35 at 3, P6=32 at 6) | 6.00 |
  | P9    | 22 | 10 (P4=10 at 12, P5=30 at 8; 2nd-NN at dist 10) | 10.00 |

  **Step 2 — Compute mutual reachability distances (consecutive sorted pairs):**

  Sorting points: 5, 8, 9, 10, 22, 30, 32, 35, 38

  | Pair         | dist | core\_dist left | core\_dist right | MRD   |
  |--------------|------|-----------------|------------------|-------|
  | P1=5, P2=8   | 3    | 4.00            | 2.00             | 4.00  |
  | P2=8, P3=9   | 1    | 2.00            | 1.00             | 2.00  |
  | P3=9, P4=10  | 1    | 1.00            | 2.00             | 2.00  |
  | P4=10, P9=22 | 12   | 2.00            | 10.00            | 12.00 |
  | P9=22, P5=30 | 8    | 10.00           | 5.00             | 10.00 |
  | P5=30, P6=32 | 2    | 5.00            | 3.00             | 5.00  |
  | P6=32, P7=35 | 3    | 3.00            | 3.00             | 3.00  |
  | P7=35, P8=38 | 3    | 3.00            | 6.00             | 6.00  |

  **Step 3 — Build the MST and cut from largest MRD:**

  The two largest MRD edges are:
  - P4–P9: MRD = 12.00 → **P9 separates first** (becomes noise)
  - P9–P5: MRD = 10.00 → **Fast and Slow groups separate**

  Cutting from largest to smallest reveals:

  ```
  Lambda=0.083 (1/12.00): P9 splits off → NOISE
  Lambda=0.100 (1/10.00): Fast group {P1,P2,P3,P4} and Slow group {P5,P6,P7,P8} separate
  ```

  **The Final Clusters:**

  ```
  Cluster 1 (Fast): {P1=5, P2=8, P3=9, P4=10}
  Cluster 2 (Slow): {P5=30, P6=32, P7=35, P8=38}
  Noise:            {P9=22}
  ```

  **Visual Analogy:**

  Imagine the delivery times on a number line as mountains of density. P2–P4 form a sharp peak (tight cluster), P5–P7 form a broader hill (looser cluster). HDBSCAN finds both mountains regardless of their different widths because it adapts to local density — unlike DBSCAN which would need `eps` tuned to one or the other. P9 sits in the valley between both mountains, too isolated to be part of either.

  > **Note:** HDBSCAN uses `min_cluster_size` instead of `eps`. A group must persist for at least `min_cluster_size` points to be extracted as a cluster. This single parameter is far more intuitive than DBSCAN's two-parameter `(eps, min_pts)` combination.

</details>

---

## HDBSCAN (n features)

HDBSCAN with multiple features computes core distances and mutual reachability distances using **Euclidean distance across all features simultaneously**, then builds the MST and condensed cluster hierarchy in the same way. Because it uses MRD rather than raw distance, it naturally handles clusters of different densities — something K-Means and DBSCAN both struggle with.

**Mutual reachability distance (n features):**

```
mrd(xi, xj) = max( core_dist(xi),  core_dist(xj),  ||xi - xj||₂ )
```

Also written as:

```
core_dist_k(xi) = d(xi, NN_k(xi))   (distance to k-th nearest neighbour)

Cluster stability:
  S(C) = sum over xi in C of  ( lambda_death(xi) - lambda_birth(C) )
```

Where `lambda_birth(C)` is the density level at which cluster C first appears, `lambda_death(xi)` is the density level at which point xi leaves the cluster, and `S(C)` is the total persistence of the cluster — higher stability means it is selected as a final cluster.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Grouping Students by Study Hours and Exam Score</summary>
  <br/>

  Imagine grouping students based on hours studied (x1) and exam score (x2), with one outlier student whose profile does not match either group.

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

  **Parameter:** `min_cluster_size = 2`

  **Step 1 — Compute core distances:**

  For each point, find the distance to its 2nd nearest neighbour:

  | Student | 2nd nearest neighbour | core\_dist |
  |---------|-----------------------|------------|
  | A1      | A3 (d=6.3246)         | 6.3246     |
  | A2      | A3 (d=3.1623)         | 3.1623     |
  | A3      | M1 (d=6.0828)         | 6.0828     |
  | B1      | B3 (d=8.2462)         | 8.2462     |
  | B2      | B3 (d=4.1231)         | 4.1231     |
  | B3      | B1 (d=8.2462)         | 8.2462     |
  | M1      | A2 (d=9.2195)         | 9.2195     |
  | N1      | A1 (d=17.6918)        | 19.2094    |

  > A1 has core\_dist = 6.3246 (not 3.1623) because its 1st nearest neighbour is A2 (d=3.16) but its **2nd** nearest neighbour is A3 (d=6.32). N1's large core\_dist (19.2094) signals it is isolated — no close neighbours.

  **Step 2 — Compute mutual reachability distances (key pairs):**

  ```
  mrd(A1, A2) = max(6.3246, 3.1623, 3.1623) = 6.3246
  mrd(A2, A3) = max(3.1623, 6.0828, 3.1623) = 6.0828
  mrd(A3, M1) = max(6.0828, 9.2195, 6.0828) = 9.2195
  mrd(A2, M1) = max(3.1623, 9.2195, 9.2195) = 9.2195
  mrd(B1, B2) = max(8.2462, 4.1231, 4.1231) = 8.2462
  mrd(B1, B3) = max(8.2462, 8.2462, 8.2462) = 8.2462
  mrd(M1, B1) = max(9.2195, 8.2462,10.1980) = 10.1980
  mrd(A1, N1) = max(6.3246,19.2094,17.6918) = 19.2094
  ```

  Note how `mrd(A1, N1) = 19.2094` — dominated by N1's own large core distance, not the raw distance. MRD effectively "pushes" isolated points far from everything else.

  **Step 3 — Build the Minimum Spanning Tree:**

  Using Prim's algorithm on the MRD matrix, the MST is:

  | Edge       | MRD     | lambda = 1/MRD |
  |------------|---------|----------------|
  | A2 -- A3   | 6.0828  | 0.1644         |
  | A1 -- A2   | 6.3246  | 0.1581         |
  | B1 -- B2   | 8.2462  | 0.1213         |
  | B1 -- B3   | 8.2462  | 0.1213         |
  | A2 -- M1   | 9.2195  | 0.1085         |
  | M1 -- B1   | 10.1980 | 0.0981         |
  | A1 -- N1   | 19.2094 | 0.0521         |

  ```mermaid
  flowchart LR
      A1 -->|mrd=6.3246| A2
      A2 -->|mrd=6.0828| A3
      A2 -->|mrd=9.2195| M1
      M1 -->|mrd=10.198| B1
      B1 -->|mrd=8.2462| B2
      B1 -->|mrd=8.2462| B3
      A1 -->|mrd=19.2094| N1

      style A1 fill:#74c0fc,stroke:#1971c2,color:#000
      style A2 fill:#74c0fc,stroke:#1971c2,color:#000
      style A3 fill:#74c0fc,stroke:#1971c2,color:#000
      style M1 fill:#74c0fc,stroke:#1971c2,color:#000
      style B1 fill:#51cf66,stroke:#2f9e44,color:#000
      style B2 fill:#51cf66,stroke:#2f9e44,color:#000
      style B3 fill:#51cf66,stroke:#2f9e44,color:#000
      style N1 fill:#ff6b6b,stroke:#c0392b,color:#fff
  ```

  **Step 4 — Cut MST from largest MRD and build the hierarchy:**

  Cutting edges from largest MRD to smallest reveals the nested cluster structure:

  ```
  Cut A1--N1  (MRD=19.2094, lambda=0.0521): N1 becomes NOISE — too isolated to join any cluster
  Cut M1--B1  (MRD=10.1980, lambda=0.0981): {A1,A2,A3,M1} and {B1,B2,B3} split into two clusters
  Cut A2--M1  (MRD=9.2195,  lambda=0.1085): M1 is a border — absorbed into Cluster A
  Cut B1--B2  (MRD=8.2462,  lambda=0.1213): B1,B2,B3 remain connected as Cluster B
  Cut B1--B3  (MRD=8.2462,  lambda=0.1213): same level — B group fully formed
  Cut A1--A2  (MRD=6.3246,  lambda=0.1581): A1,A2,A3 tighten into Cluster A core
  Cut A2--A3  (MRD=6.0828,  lambda=0.1644): innermost A connection — most stable pair
  ```

  **Step 5 — Compute cluster stability and select final clusters:**

  Stability `S(C)` = sum of `(lambda_death - lambda_birth)` for each point in the cluster. `lambda_birth` is the level at which the cluster first appears; `lambda_death(xi)` is the level at which point xi would leave.

  For **Cluster A** (born when M1--B1 cut at lambda=0.0981):

  | Point | lambda\_death | lambda\_birth | Contribution |
  |-------|--------------|---------------|--------------|
  | A1    | 0.0808       | 0.0981        | −0.0172      |
  | A2    | 0.1085       | 0.0981        | +0.0104      |
  | A3    | 0.1085       | 0.0981        | +0.0104      |
  | M1    | 0.0808       | 0.0981        | −0.0172      |

  `S(Cluster A) = −0.0136`

  For **Cluster B** (born at same split, lambda=0.0981):

  | Point | lambda\_death | lambda\_birth | Contribution |
  |-------|--------------|---------------|--------------|
  | B1    | 0.1213       | 0.0981        | +0.0232      |
  | B2    | 0.1213       | 0.0981        | +0.0232      |
  | B3    | 0.1213       | 0.0981        | +0.0232      |

  `S(Cluster B) = +0.0696`

  Cluster B has high positive stability — its points survive well above their birth lambda. Cluster A has near-zero stability because A1 and M1 have large core distances and die out quickly.

  **The Final Clusters:**

  ```
  Cluster 1 (Low effort): {A1, A2, A3, M1}   stability = −0.0136  (selected — no better split)
  Cluster 2 (High effort): {B1, B2, B3}       stability = +0.0696  (selected — persistent)
  Noise: {N1}                                  (never reached any cluster)
  ```

  **Step 6 — Prediction Example:**

  A new student studied 3.5 hours and scored 57. Which cluster?

  Compute approximate MRD to each training point:

  ```
  d(new, A3=(4,58))  = sqrt(0.25 + 1)   = 1.1180  mrd ≈ max(1.1180, 6.0828) = 6.0828
  d(new, A2=(3,55))  = sqrt(0.25 + 4)   = 2.0616  mrd ≈ max(2.0616, 3.1623) = 3.1623
  d(new, A1=(2,52))  = sqrt(2.25 + 25)  = 5.2202  mrd ≈ max(5.2202, 6.3246) = 6.3246
  d(new, M1=(5,64))  = sqrt(2.25 + 49)  = 7.1589  mrd ≈ max(7.1589, 9.2195) = 9.2195
  d(new, B1=(7,74))  = sqrt(12.25+289)  = 17.3566 mrd ≈ max(17.3566, 8.2462) = 17.3566
  d(new, N1=(15,40)) = sqrt(132.25+289) = 20.5244 mrd ≈ max(20.5244,19.2094) = 20.5244
  ```

  The new point has the smallest MRD to A2, A3, and A1 — all members of Cluster 1. It is density-reachable from Cluster 1 at a much lower lambda than Cluster 2.

  ```
  → Assigned to Cluster 1 (Low effort group)
  ```

  **Interpreting the result:**

  HDBSCAN correctly identifies both groups even though they have different densities (A-group is tighter than B-group). N1 is flagged as noise. M1, the bridge point, is absorbed into Cluster A because its MRD to A3 (9.2195) is lower than its MRD to B1 (10.1980) — it sits closer to the A-group in density-adjusted space.

  > **Note:** HDBSCAN also produces **soft cluster membership probabilities** for each point — a number between 0 and 1 indicating how confidently the point belongs to its assigned cluster. Core points deep inside a cluster score near 1; border points and points near the noise boundary score lower.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

HDBSCAN is more robust than DBSCAN but inherits some of the same challenges. It can struggle with **extremely high-dimensional data** (the curse of dimensionality makes all pairwise distances converge), very **large datasets** (the MST computation is O(n² log n) in the naive case, though approximate algorithms exist), and datasets where clusters differ so drastically in density that no single `min_cluster_size` captures all of them. It also does not handle **streaming or incremental data** well — new points require re-running the full algorithm. In such cases, alternatives include DBSCAN (simpler, faster, sufficient when density is uniform), K-Means (when cluster count is known and shapes are roughly spherical), OPTICS (produces a reachability plot rather than committing to one set of clusters), or Gaussian Mixture Models (probabilistic soft assignments with an explicit generative model).

---

## Error and the Cost Function

### Why HDBSCAN Has No Traditional Cost Function

Like DBSCAN, HDBSCAN does not optimize a differentiable loss function. It is a **hierarchical rule-based algorithm**: given fixed `min_cluster_size`, the MST, dendrogram, and stability scores are all deterministic. There are no weights, gradients, or iterative parameter updates.

Cluster quality is assessed after the fact using external metrics:

| Metric                    | Definition                                                                        | Range    |
|---------------------------|-----------------------------------------------------------------------------------|----------|
| Silhouette Score          | Mean ratio of intra-cluster cohesion to inter-cluster separation                  | −1 to +1 |
| Membership probability    | HDBSCAN-native score: how confidently each point belongs to its cluster           | 0 to 1   |
| Noise ratio               | Fraction of points labeled as noise                                               | 0 to 1   |
| DBCV (Density-Based CV)   | Validity index designed for density-based clusters; accounts for varying density  | −1 to +1 |

---

### Objective: Extract the Most Stable Clusters

The algorithm's goal is to build a full hierarchy of density-based clusters from the MST and then select the set of non-overlapping clusters with the **maximum total stability** — those that persist longest as the density threshold increases.

---

### Core Concept: Mutual Reachability and Stability

HDBSCAN builds its hierarchy through three connected ideas:

**Mutual Reachability Distance (MRD)** — smooths raw distances by respecting local density:

$$mrd(x_i, x_j) = \max\bigl(\text{core\_dist}(x_i),\ \text{core\_dist}(x_j),\ d(x_i, x_j)\bigr)$$

**Lambda (density level)** — the inverse of MRD, representing the density threshold at which two points are considered connected:

$$\lambda(x_i, x_j) = \frac{1}{mrd(x_i, x_j)}$$

**Cluster Stability** — the total persistence of all points in a cluster from its birth to each point's departure:

$$S(C) = \sum_{x_i \in C} \bigl(\lambda_{\text{death}}(x_i) - \lambda_{\text{birth}}(C)\bigr)$$

Where `lambda_birth(C)` is the density level at which cluster C first appears in the hierarchy and `lambda_death(xi)` is the density level at which point xi falls out of (or below) the cluster.

**Example (from the 2-feature dataset, Cluster B = {B1,B2,B3}):**

```
lambda_birth(Cluster B) = 1 / mrd(M1,B1) = 1 / 10.1980 = 0.0981
(Cluster B is born when the M1--B1 edge is cut, separating it from Cluster A)

lambda_death(B1) = 1 / max_intra_mrd(B1) = 1 / 8.2462 = 0.1213
lambda_death(B2) = 1 / max_intra_mrd(B2) = 1 / 8.2462 = 0.1213
lambda_death(B3) = 1 / max_intra_mrd(B3) = 1 / 8.2462 = 0.1213

S(Cluster B) = (0.1213−0.0981) + (0.1213−0.0981) + (0.1213−0.0981)
             = 0.0232 + 0.0232 + 0.0232
             = 0.0696
```

A stability of 0.0696 means Cluster B survives 0.0696 units of lambda above its birth level — a meaningfully persistent, dense group.

---

### Effect of min\_cluster\_size

The single parameter `min_cluster_size` controls what counts as a real cluster versus noise. It is more forgiving than DBSCAN's `(eps, min_pts)` pair because HDBSCAN adapts to local density — the same `min_cluster_size` works across clusters of different densities.

**Effect on the 2-feature student dataset:**

| min\_cluster\_size | Clusters found | Noise points | Behaviour                                               |
|--------------------|---------------|--------------|--------------------------------------------------------|
| 2                  | 2             | 1            | Both groups found; N1 is noise; M1 absorbed into Cluster A |
| 3                  | 2             | 1            | Both groups survive; stability scores shift slightly    |
| 5                  | 1             | 3            | Only the larger/denser group qualifies; smaller one becomes noise |
| 8                  | 0             | 8            | min_cluster_size = n: everything is noise              |

A rule of thumb: set `min_cluster_size` to the smallest meaningful group size in your domain — for customer segments, perhaps 50; for anomaly detection, perhaps 5.

---

### Alternative Notation (Membership Probability)

Unlike DBSCAN (which gives hard 0/1 cluster labels), HDBSCAN natively produces **soft membership probabilities** — a number between 0 and 1 for each point indicating how confident the algorithm is in the assignment:

$$P(x_i \in C) = \frac{\lambda_{\text{death}}(x_i) - \lambda_{\text{birth}}(C)}{\lambda_{\text{death}}(x_i)}$$

**Example (point A2=(3,55) in Cluster A):**

```
lambda_birth(Cluster A) = 0.0981
lambda_death(A2)        = 0.1085

P(A2 in Cluster A) = (0.1085 - 0.0981) / 0.1085
                   = 0.0104 / 0.1085
                   = 0.0959
```

**Example (point B2=(8,78) in Cluster B):**

```
lambda_birth(Cluster B) = 0.0981
lambda_death(B2)        = 0.1213

P(B2 in Cluster B) = (0.1213 - 0.0981) / 0.1213
                   = 0.0232 / 0.1213
                   = 0.1913
```

B2 has a higher membership probability than A2 — it sits deeper inside its cluster relative to the birth level, making it a more confident assignment. Points with probability near 0 are essentially border/noise points that barely qualified.

---

## How Do We Find the Clusters?

HDBSCAN finds clusters through four deterministic steps — no iterative optimization required. Given a fixed `min_cluster_size`:

1. Compute core distances for all points.
2. Compute all pairwise MRDs and build the MST.
3. Cut the MST from the largest edge downward to build the condensed cluster hierarchy.
4. Compute stability for every cluster in the hierarchy; select the non-overlapping set that maximizes total stability.

---

## HDBSCAN Algorithm

The algorithm processes the MST hierarchy top-down, extracting stable clusters.

**Pseudocode:**

```
Step 1 — Core distances:
  for each xi:
      core_dist(xi) = distance to min_cluster_size-th nearest neighbour

Step 2 — Mutual reachability distances:
  for each pair (xi, xj):
      mrd(xi, xj) = max(core_dist(xi), core_dist(xj), d(xi, xj))

Step 3 — Minimum Spanning Tree:
  build MST on the MRD-weighted complete graph
  (Prim's or Kruskal's algorithm)

Step 4 — Condense hierarchy:
  sort MST edges by MRD descending
  for each edge in order:
      cut the edge → creates a split in the hierarchy
      if a resulting component has < min_cluster_size points:
          label those points as noise at this level
      else:
          record component as a cluster candidate with its birth lambda

Step 5 — Extract stable clusters:
  for each cluster candidate C:
      S(C) = sum over xi in C of (lambda_death(xi) - lambda_birth(C))
  select the non-overlapping set of clusters maximizing total stability
  label unselected points as NOISE
```

**Example trace (2-feature dataset, key cuts):**

```
Sort MST edges by MRD descending:
  A1--N1   MRD=19.2094  lambda=0.0521
  M1--B1   MRD=10.1980  lambda=0.0981
  A2--M1   MRD= 9.2195  lambda=0.1085
  B1--B2   MRD= 8.2462  lambda=0.1213
  B1--B3   MRD= 8.2462  lambda=0.1213
  A1--A2   MRD= 6.3246  lambda=0.1581
  A2--A3   MRD= 6.0828  lambda=0.1644

Cut A1--N1: N1 component has 1 point < min_cluster_size=2 → N1 = NOISE
Cut M1--B1: splits into {A1,A2,A3,M1} and {B1,B2,B3}
            both >= 2 → two cluster candidates born at lambda=0.0981
Remaining cuts tighten within each cluster — no further splits above min_cluster_size
Compute stability → select both clusters → final output
```

- A **small min\_cluster\_size** finds many small clusters and labels few points as noise.
- A **large min\_cluster\_size** requires larger groups — fewer clusters, more noise points.
- HDBSCAN is **robust to the choice** within a reasonable range because stability selection automatically rejects ephemeral micro-clusters.

---

## Summary of Key Formulas

| Concept                       | Formula                                                                            |
|-------------------------------|------------------------------------------------------------------------------------|
| Core distance                 | core\_dist(xi) = d(xi, k-th nearest neighbour), k = min\_cluster\_size             |
| Mutual reachability distance  | mrd(xi, xj) = max(core\_dist(xi), core\_dist(xj), d(xi, xj))                      |
| Lambda (density level)        | lambda(xi, xj) = 1 / mrd(xi, xj)                                                  |
| Cluster stability             | S(C) = sum\_{xi in C} (lambda\_death(xi) - lambda\_birth(C))                       |
| Membership probability        | P(xi in C) = (lambda\_death(xi) - lambda\_birth(C)) / lambda\_death(xi)            |
| Euclidean distance            | d(xi, xj) = sqrt( sum\_f (xi\_f - xj\_f)^2 )                                      |
| Cluster selection             | argmax over non-overlapping sets of candidates: sum of S(C)                        |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- HDBSCAN — Scikit-learn - *(coming soon)*

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=dGsxd67IFiU" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/dGsxd67IFiU/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=RDZUdRSDOok" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/RDZUdRSDOok/hqdefault.jpg"/>
  </a>
</div>