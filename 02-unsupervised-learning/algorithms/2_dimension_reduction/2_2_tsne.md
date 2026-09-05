# t-SNE — t-Distributed Stochastic Neighbour Embedding <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

t-SNE (1 feature) | t-SNE (n features) | Perplexity and KL Divergence | Heavy Tails

---

## What is t-SNE? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

t-SNE (t-Distributed Stochastic Neighbour Embedding) is an **unsupervised learning algorithm** used to reduce high-dimensional data to 2 or 3 dimensions for **visualization**, preserving the local neighbourhood structure of the original data.

At its core, t-SNE converts pairwise distances into probabilities — high probability for close neighbours, low probability for distant points — and then finds a low-dimensional embedding where the same probability structure is reproduced as faithfully as possible. It uses a **Gaussian distribution** in the high-dimensional space and a **Student t-distribution** (with heavier tails) in the low-dimensional space, which prevents the crowding problem that collapses all points to the centre when compressing many dimensions into two.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to find a 2D (or 3D) layout where nearby points in the original space remain nearby, and distant points are pushed apart. Four concepts are central:

- **High-dimensional similarities (P)** — pairwise probabilities computed from Gaussian kernels in the original feature space. P(j|i) is high when points i and j are close neighbours, and near zero when they are far apart.
- **Low-dimensional similarities (Q)** — pairwise probabilities computed from the Student t-distribution in the 2D embedding. Q(i,j) is high when the embedded points are close, low when they are spread apart.
- **KL divergence** — the cost function measuring how different P and Q are. Gradient descent minimizes KL(P||Q), pulling the embedding toward a layout where Q matches P.
- **Perplexity** — a hyperparameter (typically 5–50) that controls the effective number of neighbours each point considers. It determines the bandwidth (sigma) of the Gaussian kernel per point via binary search.

**Two key hyperparameters:**

```
perplexity    — effective number of neighbours (default 30); controls local vs global balance
n_iter        — number of gradient descent iterations (default 1000)
```

**High-dimensional similarity (conditional):**

```
P(j|i) = exp( -||xi - xj||^2 / (2 * sigma_i^2) )
          / sum_{k != i} exp( -||xi - xk||^2 / (2 * sigma_i^2) )
```

**Symmetrized joint probability:**

```
P(i,j) = ( P(j|i) + P(i|j) ) / (2 * m)
```

Where `sigma_i` is chosen per point so that its neighbourhood distribution has the desired perplexity, and `m` is the total number of data points.

---

## t-SNE (1 feature)

t-SNE applied to a single feature demonstrates the core idea of converting distances to probabilities before extending to the multi-feature case. With one feature, pairwise distances are just absolute differences; Gaussian similarities decay as points grow farther apart.

**Conditional similarity (1D):**

```
P(j|i) = exp( -(xi - xj)^2 / (2 * sigma^2) )
          / sum_{k != i} exp( -(xi - xk)^2 / (2 * sigma^2) )
```

Also written as:

```
P(j|i) = Gaussian( d(i,j)^2 ) / Z_i

where  d(i,j)^2 = (xi - xj)^2
       Z_i      = sum_{k != i} Gaussian( d(i,k)^2 )   (normalising constant)
```

Where:

- `xi` — value of point i in the original 1D space
- `sigma` — the bandwidth of the Gaussian kernel; larger sigma = broader neighbourhood
- `P(j|i)` — how much point i "notices" point j as a neighbour; high for nearby points
- `Z_i` — the row-normalising constant ensuring all conditional probabilities for point i sum to 1
- `d(i,j)^2` — squared distance between points i and j

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Computing Neighbourhood Similarities for Study Hours</summary>
  <br/>

  Imagine computing how strongly each student "notices" every other student as a neighbour, based purely on hours studied — before any embedding takes place.

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

  **Parameter:** `sigma = 2.0` (fixed for illustration; in practice set by perplexity)

  **Step 1 — Compute squared distances from S1 to all others:**

  ```
  d(S1,S2)^2 = (2−3)^2  = 1.00
  d(S1,S3)^2 = (2−4)^2  = 4.00
  d(S1,S4)^2 = (2−7)^2  = 25.00
  d(S1,S5)^2 = (2−8)^2  = 36.00
  d(S1,S6)^2 = (2−9)^2  = 49.00
  d(S1,S7)^2 = (2−5)^2  = 9.00
  ```

  **Step 2 — Apply Gaussian kernel with sigma=2.0:**

  ```
  exp( -d^2 / (2*2^2) ) = exp( -d^2 / 8 )

  exp(−1.00/8) = exp(−0.125) = 0.8825
  exp(−4.00/8) = exp(−0.500) = 0.6065
  exp(−25.0/8) = exp(−3.125) = 0.0439
  exp(−36.0/8) = exp(−4.500) = 0.0111
  exp(−49.0/8) = exp(−6.125) = 0.0022
  exp(−9.00/8) = exp(−1.125) = 0.3247
  ```

  **Step 3 — Normalise to get conditional probabilities P(j|S1):**

  ```
  Z(S1) = 0.8825 + 0.6065 + 0.0439 + 0.0111 + 0.0022 + 0.3247 = 1.8709

  P(S2|S1) = 0.8825 / 1.8709 = 0.4717   ← nearest neighbour, highest similarity
  P(S3|S1) = 0.6065 / 1.8709 = 0.3242
  P(S7|S1) = 0.3247 / 1.8709 = 0.1735
  P(S4|S1) = 0.0439 / 1.8709 = 0.0235
  P(S5|S1) = 0.0111 / 1.8709 = 0.0059
  P(S6|S1) = 0.0022 / 1.8709 = 0.0012   ← most distant, lowest similarity
  ```

  **Interpretation:**

  S1 (2 hours) notices S2 (3 hours) most strongly (P=0.4717) and barely notices S6 (9 hours) at all (P=0.0012). The Gaussian kernel converts distance into a smooth probability that falls off rapidly — this is what t-SNE uses to encode "who is whose neighbour" before building the 2D embedding.

  **Visual Analogy:**

  Imagine each student surrounded by a Gaussian bell curve centred on their hours. The overlap between two students' bells is proportional to their similarity. Wide bells (large sigma / high perplexity) catch more neighbours; narrow bells (low perplexity) focus only on the very closest points.

  > **Note:** In practice, sigma is not fixed — it is chosen separately for each point via binary search so that the resulting conditional distribution has the desired perplexity. Points in dense regions get a smaller sigma (tight neighbourhood); points in sparse regions get a larger sigma (wider neighbourhood).

</details>

---

## t-SNE (n features)

t-SNE with multiple features computes pairwise similarities using **Euclidean distance across all features simultaneously** (on standardized data), then iteratively finds a 2D embedding by minimizing the KL divergence between the high-dimensional similarity matrix P and the low-dimensional similarity matrix Q. The Student t-distribution in Q has heavier tails than the Gaussian in P, which prevents the crowding problem and allows well-separated clusters to emerge naturally.

**Low-dimensional similarity (Student t-kernel):**

```
Q(i,j) = (1 + ||yi - yj||^2)^(-1)
          / sum_{k != l} (1 + ||yk - yl||^2)^(-1)
```

Also written as:

```
KL(P || Q) = sum_{i,j} P(i,j) * log( P(i,j) / Q(i,j) )

Gradient:
  dC/dyi = 4 * sum_j (P(i,j) − Q(i,j)) * (yi − yj) * (1 + ||yi−yj||^2)^(-1)
```

Where `yi` is the 2D position of point i in the embedding, `Q(i,j)` uses the Student t-distribution with 1 degree of freedom (which has the heaviest useful tail), and the gradient pulls yi toward neighbours with P > Q and pushes it away from points with P < Q.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Embedding Student Data from 2 Features into 1D</summary>
  <br/>

  Imagine reducing two features — hours studied (x1) and exam score (x2) — into a single dimension that preserves which students are similar to which.

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

  **Step 1 — Standardize each feature:**

  t-SNE is sensitive to scale, so features are standardized (z-score) before computing distances:

  ```
  mean(hours) = 5.4286,  std(hours) = 2.4398
  mean(score) = 66.1429, std(score) = 10.9503

  S1 standardized: [(2−5.4286)/2.4398, (52−66.1429)/10.9503] = [−1.4045, −1.2843]
  S4 standardized: [(7−5.4286)/2.4398, (74−66.1429)/10.9503] = [+0.6437, +0.7135]
  ```

  Full standardized data:

  | Student | hours\_std | score\_std |
  |---------|------------|------------|
  | S1      | −1.4045    | −1.2843    |
  | S2      | −0.9948    | −1.0119    |
  | S3      | −0.5852    | −0.7394    |
  | S4      | +0.6437    | +0.7135    |
  | S5      | +1.0534    | +1.0767    |
  | S6      | +1.4630    | +1.4400    |
  | S7      | −0.1756    | −0.1946    |

  **Step 2 — Compute pairwise squared Euclidean distances:**

  ```
  D2(S1,S2) = (−1.4045−(−0.9948))^2 + (−1.2843−(−1.0119))^2
            = (−0.4097)^2 + (−0.2724)^2
            = 0.1679 + 0.0742 = 0.2420

  D2(S1,S4) = (−1.4045−0.6437)^2 + (−1.2843−0.7135)^2
            = (−2.0482)^2 + (−1.9978)^2
            = 4.1951 + 3.9912 = 8.1865

  D2(S4,S5) = (0.6437−1.0534)^2 + (0.7135−1.0767)^2
            = (−0.4097)^2 + (−0.3632)^2
            = 0.1679 + 0.1319 = 0.2998
  ```

  Key distances (all pairs):

  | Pair   | D2(i,j) | Interpretation                     |
  |--------|---------|------------------------------------|
  | S1,S2  | 0.2420  | Very close — similar students      |
  | S2,S3  | 0.2420  | Very close — similar students      |
  | S4,S5  | 0.2998  | Very close — similar students      |
  | S5,S6  | 0.2998  | Very close — similar students      |
  | S3,S7  | 0.4647  | Fairly close — S7 bridges groups   |
  | S1,S4  | 8.1865  | Far apart — different groups       |
  | S1,S6  | 15.6443 | Most distant pair                  |

  **Step 3 — Compute high-dim similarities P with sigma=1.0:**

  For P(S2|S1): `exp(−0.2420 / 2) = exp(−0.1210) = 0.8861`

  For P(S4|S1): `exp(−8.1865 / 2) = exp(−4.0932) = 0.0167`

  After normalising and symmetrizing across all 7 points:

  ```
  P(S1,S2) = 0.0623   ← close neighbours, high joint probability
  P(S2,S3) = 0.0519   ← close neighbours
  P(S4,S5) ≈ 0.0679   ← high similarity within group B
  P(S1,S4) = 0.0012   ← far apart, low joint probability
  ```

  **Step 4 — Initialise the 1D embedding and compute Q:**

  Start with random 1D positions, then run gradient descent. After some iterations, suppose:

  ```
  y = [−1.5,  −1.0,  −0.5,  +0.8,  +1.2,  +1.6,  0.0]
         S1     S2     S3     S4     S5     S6     S7
  ```

  Compute Student t-kernel Q for each pair using `(1 + |yi−yj|^2)^(-1)`:

  ```
  Q_raw(S1,S2): (1 + |−1.5−(−1.0)|^2)^-1 = (1 + 0.25)^-1 = 0.8000
  Q_raw(S1,S4): (1 + |−1.5−0.8|^2)^-1    = (1 + 5.29)^-1 = 0.1590
  Q_raw(S4,S5): (1 + |0.8−1.2|^2)^-1     = (1 + 0.16)^-1 = 0.8621

  Z = sum of all Q_raw = 18.1311

  Q(S1,S2) = 0.8000 / 18.1311 = 0.0441
  Q(S1,S4) = 0.1590 / 18.1311 = 0.0088
  Q(S4,S5) = 0.8621 / 18.1311 = 0.0475
  ```

  **Step 5 — Compute KL divergence:**

  ```
  KL(P||Q) = sum_{i,j} P(i,j) * log( P(i,j) / Q(i,j) )

  Top contributions:
    (S5,S6): P=0.0720  Q=0.0475  → 0.0720*log(0.0720/0.0475) = +0.0298
    (S1,S2): P=0.0623  Q=0.0441  → 0.0623*log(0.0623/0.0441) = +0.0215
    (S1,S3): P=0.0422  Q=0.0276  → 0.0422*log(0.0422/0.0276) = +0.0179

  Total KL divergence = 0.1445
  ```

  **Step 6 — Compute gradient and update positions:**

  For S1 (y = −1.5):

  ```
  dC/dy[S1] = 4 * sum_j (P(S1,j) − Q(S1,j)) * (y[S1]−y[j]) * (1+|y[S1]−y[j]|^2)^-1

  j=S2: (0.0623−0.0441)*(−0.5)*(0.8000) = −0.0292
  j=S3: (0.0422−0.0276)*(−1.0)*(0.5000) = −0.0292
  j=S4: (0.0012−0.0088)*(−2.3)*(0.1590) = +0.0110
  j=S7: (0.0062−0.0170)*(−1.5)*(0.3077) = −0.0027

  Total gradient = 4 * (−0.0089) = −0.0356

  y[S1]_new = −1.5000 − 200 * (−0.0356) = −1.5000 + 7.12  [early exaggeration step]
  ```

  **Step 7 — The converged embedding:**

  After ~1000 iterations the embedding converges. The A-group (S1–S3) clusters to the left, the B-group (S4–S6) clusters to the right, and S7 (the bridge point) settles near the centre.

  ```mermaid
  flowchart LR
      A["Standardize features - zero mean, unit variance"]
      B["Compute pairwise D2 in high-dim space"]
      C["Gaussian similarities P - set sigma per point via perplexity"]
      D["Initialize random 2D embedding y"]
      E["Student t-kernel Q from current y"]
      F["KL divergence = sum P*log P/Q"]
      G["Gradient descent - update y to minimize KL"]
      H["Converged 2D layout - clusters visible"]

      A --> B --> C --> D --> E --> F --> G --> E
      G --> H

      style A fill:#f0c040,stroke:#b8860b,color:#000
      style B fill:#f7b731,stroke:#e67e00,color:#000
      style C fill:#74c0fc,stroke:#1971c2,color:#000
      style D fill:#74c0fc,stroke:#1971c2,color:#000
      style E fill:#51cf66,stroke:#2f9e44,color:#000
      style F fill:#51cf66,stroke:#2f9e44,color:#000
      style G fill:#ff6b6b,stroke:#c0392b,color:#fff
      style H fill:#a9e34b,stroke:#5c940d,color:#000
  ```

  **Interpreting the result:**

  The KL divergence pushes S1, S2, S3 together (their P values are high and the algorithm needs Q to match) and pulls them away from S4–S6 (whose P values are low). S7 — with moderate similarity to both groups — settles between them. The final 2D positions reflect neighbourhood structure, not absolute distances.

  > **Note:** t-SNE is **not** designed for dimensionality reduction before modelling (use PCA for that). Its output is for **visualization only** — distances in the t-SNE plot do not preserve the global structure of the original space, and cluster sizes and inter-cluster distances are not directly interpretable.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

t-SNE is designed purely for **visualization** — the output is a 2D or 3D layout that is not suitable for downstream modelling, feature extraction, or reconstruction. It is **non-deterministic**: different random initializations produce different (though structurally similar) embeddings. It does **not preserve global structure** — distances between clusters in the t-SNE plot are not meaningful; only within-cluster proximity is reliable. It scales poorly to **very large datasets** (O(n²) pairwise comparisons) unless approximate algorithms (Barnes-Hut t-SNE) are used, reducing it to O(n log n). The result is also highly sensitive to **perplexity** — wrong values can either split natural clusters or merge distinct ones. Finally, t-SNE **cannot transform new points** without re-running the full optimization. In such cases, alternatives include UMAP (faster, better global structure preservation, supports out-of-sample transforms), PCA (linear, fast, reconstructable, suitable for preprocessing), Parametric t-SNE (neural network-based, supports new point transforms), or Kernel PCA (non-linear, but with a fixed kernel rather than learned layout).

---

## Error and the Cost Function

### Why t-SNE Uses KL Divergence

t-SNE does not minimize reconstruction error like PCA. Instead, it minimizes the **KL divergence** between the high-dimensional probability distribution P and the low-dimensional probability distribution Q — a measure of how poorly Q approximates P.

| Scenario                            | KL contribution              | Effect on embedding              |
|-------------------------------------|------------------------------|----------------------------------|
| P(i,j) large, Q(i,j) small         | Large positive → high cost   | Push yi and yj closer together   |
| P(i,j) small, Q(i,j) large         | Near zero (P≈0 so term≈0)    | Weak repulsion — can stay apart  |
| P(i,j) ≈ Q(i,j)                    | Near zero → low cost         | Embedding matches similarity     |

**Why not symmetric divergence?**

KL(P||Q) is asymmetric: it penalizes placing far-apart points close together in the embedding (P small, Q large) much less than it penalizes placing close points far apart (P large, Q small). This means t-SNE prioritizes preserving **close neighbourhoods** over preserving global distances — tight clusters will always appear, even if inter-cluster distances are distorted.

---

### Objective: Minimize KL Divergence Between P and Q

The algorithm's goal is to find 2D positions `y_1, ..., y_m` for all points such that the Student t-distribution similarities Q(i,j) in the low-dimensional space match the Gaussian similarities P(i,j) from the high-dimensional space as closely as possible.

---

### Core Concept: Heavy Tails Solve the Crowding Problem

The key insight of t-SNE over earlier methods is using the **Student t-distribution** (with 1 degree of freedom) instead of a Gaussian in the low-dimensional space.

**Why heavy tails matter:**

In high dimensions, a Gaussian with moderate sigma gives reasonable neighbour probabilities across a wide range of distances. But when projecting to 2D, there is simply not enough room to place all points correctly using a Gaussian — moderately-distant points crowd into the centre. The Student t-distribution has much heavier tails: it assigns much larger Q values to moderately distant points than a Gaussian would, creating natural repulsion that spreads the embedding out.

**Comparison at different distances:**

| Distance \|y\| | Gaussian kernel | Student-t kernel | Ratio (t/G) |
|---------------|-----------------|------------------|-------------|
| 1.0           | 0.6065          | 0.5000           | 0.82        |
| 2.0           | 0.1353          | 0.2000           | 1.48        |
| 3.0           | 0.0111          | 0.1000           | 9.00        |
| 5.0           | 0.0000035       | 0.0385           | 10,321      |
| 10.0          | ≈ 0             | 0.0099           | enormous    |

At distance 3, the Student-t kernel is 9× larger than the Gaussian. This means Q(i,j) stays much higher for moderately distant pairs, requiring P(i,j) to also be high for the pair to stay close — natural repulsion between dissimilar points.

---

### Cost Function: KL Divergence

**Formula:**

$$KL(P \| Q) = \sum_{i \neq j} P(i,j) \cdot \log \frac{P(i,j)}{Q(i,j)}$$

Where:
- `P(i,j)` — symmetrized joint probability in the high-dimensional space
- `Q(i,j)` — joint probability under the Student t-kernel in the low-dimensional space
- The sum runs over all ordered pairs (i,j) with i ≠ j

**Example calculation (from the 2-feature dataset):**

```
P(S1,S2) = 0.0623,  Q(S1,S2) = 0.0441
→ contribution: 0.0623 * log(0.0623/0.0441) = 0.0623 * 0.3459 = 0.0215

P(S5,S6) = 0.0720,  Q(S5,S6) = 0.0475
→ contribution: 0.0720 * log(0.0720/0.0475) = 0.0720 * 0.4149 = 0.0298

Total KL divergence = 0.1445
```

A perfectly converged embedding would have Q = P everywhere and KL = 0. In practice, the minimum is non-zero because it is impossible to perfectly preserve all high-dimensional relationships in 2D.

---

### Alternative Notation — Effect of Perplexity

Since t-SNE has no components or eigenvalues to select from, the **perplexity** is the key tuning dial. It sets the effective number of neighbours via the Shannon entropy of each row of P:

$$\text{Perplexity}_i = 2^{H(P_i)} \quad \text{where} \quad H(P_i) = -\sum_{j \neq i} P(j|i) \log_2 P(j|i)$$

**Perplexity and its effect on the embedding:**

| Perplexity | Effective neighbours | Behaviour                                             |
|------------|---------------------|-------------------------------------------------------|
| 5          | ~5                  | Very local — tight micro-clusters, large voids        |
| 30         | ~30                 | Balanced — default, good for most datasets            |
| 50         | ~50                 | Smoother — more global structure visible              |
| 100        | ~100                | Over-smoothed — distinct clusters may merge           |

**Perplexity vs sigma (from the 1-feature example):**

| sigma | Shannon entropy H | Perplexity = 2^H |
|-------|-------------------|-----------------|
| 0.5   | 0.7490            | 1.68            |
| 1.0   | 1.5172            | 2.86            |
| 2.0   | 2.3190            | 4.99            |

Larger sigma → flatter distribution → higher entropy → higher perplexity → more neighbours considered. The binary search during t-SNE fitting adjusts sigma per point until the entropy matches the target perplexity.

---

## How Do We Find the Embedding?

t-SNE finds the embedding through **iterative gradient descent** on the KL divergence — not a closed-form solution. Each iteration computes the gradient of KL(P||Q) with respect to every embedded point's position, then updates all positions simultaneously.

---

## t-SNE Algorithm

The algorithm runs in five iterative steps per gradient descent update.

**Pseudocode:**

```
Step 1 — Standardize input:
  X_std = (X - mean) / std   for each feature

Step 2 — Compute high-dim similarities P:
  for each point i:
      find sigma_i via binary search so that Perplexity(P_i) = target_perplexity
      P(j|i) = exp(−D2(i,j)/(2*sigma_i^2)) / sum_{k!=i} exp(−D2(i,k)/(2*sigma_i^2))
  P(i,j) = (P(j|i) + P(i|j)) / (2*m)   (symmetrize)

Step 3 — Initialise embedding:
  y_i ~ Normal(0, 0.0001)   for all i   (tiny random positions)

Step 4 — Gradient descent (repeat n_iter times):
  compute Q(i,j) = (1 + ||yi−yj||^2)^-1 / Z
  compute gradient:
    dC/dyi = 4 * sum_j (P(i,j) − Q(i,j)) * (yi − yj) * (1 + ||yi−yj||^2)^-1
  update positions:
    yi_new = yi − learning_rate * gradient + momentum * (yi − yi_old)

Step 5 — Return final y positions (2D or 3D layout)
```

**Example trace (2-feature dataset, one gradient step for S1):**

```
Standardize: S1 = [−1.4045, −1.2843]

P(S1,S2) = 0.0623  (close neighbours)
P(S1,S4) = 0.0012  (distant — different group)

Init: y[S1] = −1.50  (hypothetical position after early iterations)

Q(S1,S2) = 0.8000 / 18.1311 = 0.0441
Q(S1,S4) = 0.1590 / 18.1311 = 0.0088

Gradient contribution from S2:
  (P−Q)*(yi−yj)*w = (0.0623−0.0441)*(−0.50)*0.8000 = −0.0073
  scaled by 4: −0.0292

Gradient contribution from S4:
  (P−Q)*(yi−yj)*w = (0.0012−0.0088)*(−2.30)*0.1590 = +0.0028
  scaled by 4: +0.0110

Total gradient dC/dy[S1] = −0.0356
y[S1]_new = −1.5000 − 200*(−0.0356) = 5.6128  [large step during early exaggeration]
```

- A **large perplexity** sees more neighbours — global structure is preserved but local clusters blur.
- A **small perplexity** focuses tightly — local clusters are sharp but global layout is unreliable.
- **Early exaggeration** (first ~250 iterations): P values are multiplied by 4–12 to encourage tight early cluster formation.
- **Learning rate** (default 200): too small → slow convergence; too large → points collapse or oscillate.

---

## Summary of Key Formulas

| Concept                          | Formula                                                                                          |
|----------------------------------|--------------------------------------------------------------------------------------------------|
| Standardization                  | x\_std = (x − mean) / std                                                                        |
| High-dim conditional similarity  | P(j\|i) = exp(−D2(i,j)/2σ²) / Σ\_k exp(−D2(i,k)/2σ²)                                           |
| Symmetrized joint probability    | P(i,j) = (P(j\|i) + P(i\|j)) / (2m)                                                             |
| Student t-kernel (low-dim)       | Q\_raw(i,j) = (1 + \|\|yi−yj\|\|²)^(−1)                                                         |
| Normalised low-dim similarity    | Q(i,j) = Q\_raw(i,j) / Σ\_{k≠l} Q\_raw(k,l)                                                     |
| KL divergence (cost function)    | KL(P\|\|Q) = Σ\_{i≠j} P(i,j) · log(P(i,j) / Q(i,j))                                            |
| Gradient                         | dC/dyi = 4 · Σ\_j (P(i,j)−Q(i,j)) · (yi−yj) · (1+\|\|yi−yj\|\|²)^(−1)                         |
| Perplexity                       | Perp\_i = 2^(H(P\_i)),  H = −Σ P(j\|i) log₂ P(j\|i)                                            |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- t-SNE *(coming soon)*

---

## Recommended Video(s) <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=NEaUSP4YerM" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/NEaUSP4YerM/hqdefault.jpg"/>
  </a>
</div>
