# Regression Trees <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

Regression Tree (1 feature) | Regression Tree (n features) | Splitting Criteria | Pruning

---

## What is a Regression Tree? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

A Regression Tree is a **supervised learning algorithm** used to predict a **continuous numeric value** (e.g., exam score, house price, temperature) by learning a sequence of **if-then-else rules** from the data.

At its core, it is identical in structure to a Decision Tree — it recursively partitions the feature space into regions using binary questions at each node. The key difference is at the leaves: instead of predicting a class label by majority vote, a Regression Tree predicts a **numeric value by averaging** all the training examples that fall into that leaf. The result is a step-function approximation of the target, where each region of the input space is assigned a constant predicted value.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to build a tree that accurately predicts continuous target values while remaining as simple as possible. Three concepts are central:

- **Node** — a decision point that tests one feature against a threshold (`x <= t`). The root node is the first split; internal nodes are subsequent splits.
- **Branch** — the outcome of a test (left branch = condition true; right branch = condition false).
- **Leaf** — a terminal node that holds the final prediction: the **mean of all target values** in that leaf's training samples.

At every node, the algorithm searches over all features and all thresholds to find the split that **reduces variance the most**. Variance reduction measures how much more homogeneous (tightly clustered around a single value) the target values become after the split.

**Splitting criteria:**

```
MSE (Mean Squared Error) of a node S:
  MSE(S) = (1/|S|) * Σ (yᵢ - ȳ)²

Variance Reduction:
  VR(S, t) = MSE(S) - |Sₗ|/|S| * MSE(Sₗ) - |Sᵣ|/|S| * MSE(Sᵣ)
```

Where `ȳ` is the mean target value in node `S`, `Sₗ` and `Sᵣ` are the left and right child nodes, and `|S|` is the number of samples. A split with high VR produces children whose target values are much more tightly grouped than the parent.

---

## Regression Tree (1 feature)

A Regression Tree with a single feature partitions the number line into intervals by choosing one or more threshold values. Each threshold creates a binary question: "Is x ≤ t?" The algorithm picks the threshold that yields the greatest reduction in MSE across the two resulting child nodes.

**Splitting rule:**

```
if x <= threshold  →  go left  (predict mean of left samples)
if x >  threshold  →  go right (predict mean of right samples)
```

Also written as:

```
split(x, t):
    left  = { yᵢ | xᵢ <= t }
    right = { yᵢ | xᵢ >  t }
    choose t that maximizes Variance Reduction (VR)
```

Where:

- `x` — the single input feature
- `t` — the threshold value being tested at the node
- `left`, `right` — the subsets of target values that fall into each child node
- `Variance Reduction` — the reduction in MSE achieved by the split
- `ȳₗ`, `ȳᵣ` — the mean target values of the left and right children (the leaf predictions)

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Exam Score Based on Study Hours</summary>
  <br/>

  Imagine predicting a student's exam score based on how many hours they studied.

  **Dataset (Hours Studied vs Exam Score):**

  | Hours Studied (x) | Exam Score (y) |
  |-------------------|----------------|
  | 1                 | 52             |
  | 2                 | 55             |
  | 3                 | 60             |
  | 4                 | 63             |
  | 5                 | 66             |
  | 6                 | 70             |
  | 7                 | 74             |
  | 8                 | 78             |
  | 9                 | 82             |
  | 10                | 85             |

  **Step 1 — Measure root node variance:**

  The root node holds all 10 examples. The baseline prediction (before any split) is the global mean:

  ```
  ȳ = (52+55+60+63+66+70+74+78+82+85) / 10 = 685 / 10 = 68.50

  Squared deviations from mean:
    (52-68.5)² = 272.25,  (55-68.5)² = 182.25,  (60-68.5)² = 72.25
    (63-68.5)² =  30.25,  (66-68.5)² =   6.25,  (70-68.5)² =  2.25
    (74-68.5)² =  30.25,  (78-68.5)² =  90.25,  (82-68.5)² = 182.25
    (85-68.5)² = 272.25

  SSR = 1140.50
  MSE(root) = 1140.50 / 10 = 114.05
  ```

  **Step 2 — Evaluate all candidate splits:**

  The algorithm tries every midpoint between consecutive feature values and computes the Variance Reduction (VR) for each:

  | Threshold      | Left node                     | Left mean | Right node                    | Right mean | VR      |
  |----------------|-------------------------------|-----------|-------------------------------|------------|---------|
  | hours <= 1.5   | {52}                          | 52.00     | {55,60,63,66,70,74,78,82,85}  | 70.33      | 30.2500 |
  | hours <= 2.5   | {52,55}                       | 53.50     | {60,63,66,70,74,78,82,85}     | 72.25      | 56.2500 |
  | hours <= 3.5   | {52,55,60}                    | 55.67     | {63,66,70,74,78,82,85}        | 74.00      | 70.5833 |
  | hours <= 4.5   | {52,55,60,63}                 | 57.50     | {66,70,74,78,82,85}           | 75.83      | 80.6667 |
  | **hours <= 5.5** | **{52,55,60,63,66}**        | **59.20** | **{70,74,78,82,85}**          | **77.80**  | **86.4900** |
  | hours <= 6.5   | {52,55,60,63,66,70}           | 61.00     | {74,78,82,85}                 | 79.75      | 84.3750 |
  | hours <= 7.5   | {52,55,60,63,66,70,74}        | 62.86     | {78,82,85}                    | 81.67      | 74.2976 |
  | hours <= 8.5   | {52,55,60,63,66,70,74,78}     | 64.75     | {82,85}                       | 83.50      | 56.2500 |
  | hours <= 9.5   | {52,55,60,63,66,70,74,78,82}  | 66.67     | {85}                          | 85.00      | 30.2500 |

  **Step 3 — Select the best split and show VR calculation:**

  The threshold `hours <= 5.5` achieves the highest VR of **86.4900**.

  ```
  Left  child (hours <= 5.5, n=5): {52,55,60,63,66}  ȳₗ = 59.20  MSE = 26.16
  Right child (hours >  5.5, n=5): {70,74,78,82,85}  ȳᵣ = 77.80  MSE = 28.96

  VR = MSE(root) - (5/10)*MSE(left) - (5/10)*MSE(right)
     = 114.05 - 0.5*26.16 - 0.5*28.96
     = 114.05 - 13.08 - 14.48
     = 86.49
  ```

  **Step 4 — Grow one more level (depth = 2):**

  Neither child is perfectly homogeneous (MSE > 0), so the tree can split further. The best second-level splits are:

  - Left child  → best split at `hours <= 2.5` (VR = 21.66)
  - Right child → best split at `hours <= 7.5` (VR = 22.43)

  ```
  LL (hours <= 2.5):          {52, 55}           ȳ = 53.50   MSE =  2.25
  LR (2.5 < hours <= 5.5):   {60, 63, 66}        ȳ = 63.00   MSE =  6.00
  RL (5.5 < hours <= 7.5):   {70, 74}            ȳ = 72.00   MSE =  4.00
  RR (hours > 7.5):           {78, 82, 85}        ȳ = 81.67   MSE =  8.22
  ```

  **Step 5 — The Fitted Tree (depth = 2):**

 ```mermaid
  flowchart TD
      A["Root: hours <= 5.5? | n=10 | MSE=114.05 | mean=68.50"]
      B["hours <= 2.5? | n=5 | MSE=26.16 | mean=59.20"]
      C["hours <= 7.5? | n=5 | MSE=28.96 | mean=77.80"]
      D["Leaf: predict 53.50 | n=2 | values: 52, 55"]
      E["Leaf: predict 63.00 | n=3 | values: 60, 63, 66"]
      F["Leaf: predict 72.00 | n=2 | values: 70, 74"]
      G["Leaf: predict 81.67 | n=3 | values: 78, 82, 85"]

      A -->|YES - hours <= 5.5| B
      A -->|NO  - hours > 5.5| C
      B -->|YES - hours <= 2.5| D
      B -->|NO  - hours > 2.5| E
      C -->|YES - hours <= 7.5| F
      C -->|NO  - hours > 7.5| G

      style A fill:#f0c040,stroke:#b8860b,color:#000
      style B fill:#f7b731,stroke:#e67e00,color:#000
      style C fill:#f7b731,stroke:#e67e00,color:#000
      style D fill:#74c0fc,stroke:#1971c2,color:#000
      style E fill:#74c0fc,stroke:#1971c2,color:#000
      style F fill:#74c0fc,stroke:#1971c2,color:#000
      style G fill:#74c0fc,stroke:#1971c2,color:#000
  ```

  **Prediction Examples:**

  | Hours | Tree path                          | Predicted | Actual | Error  |
  |-------|------------------------------------|-----------|--------|--------|
  | 2     | <= 5.5 → YES → <= 2.5 → YES       | 53.50     | 55     | −1.50  |
  | 5     | <= 5.5 → YES → <= 2.5 → NO        | 63.00     | 66     | −3.00  |
  | 7     | <= 5.5 → NO  → <= 7.5 → YES       | 72.00     | 74     | −2.00  |
  | 9     | <= 5.5 → NO  → <= 7.5 → NO        | 81.67     | 82     | −0.33  |

  **Visual Analogy:**

  Imagine plotting students on a number line by hours studied, with their exam scores on the vertical axis. The regression tree draws vertical dividing lines at 2.5, 5.5, and 7.5 hours — splitting the data into four segments. Each segment's prediction is the flat horizontal line at the average score of the students in that segment. The result is a step-function that approximates the true relationship.

  ```mermaid
  flowchart LR
      R1["hours <= 2.5 | Leaf LL | predict 53.50 | values: 52, 55"]
      R2["2.5 < hours <= 5.5 | Leaf LR | predict 63.00 | values: 60, 63, 66"]
      R3["5.5 < hours <= 7.5 | Leaf RL | predict 72.00 | values: 70, 74"]
      R4["hours > 7.5 | Leaf RR | predict 81.67 | values: 78, 82, 85"]

      R1 --> R2 --> R3 --> R4

      style R1 fill:#dbe4ff,stroke:#4263eb,color:#000
      style R2 fill:#74c0fc,stroke:#1971c2,color:#000
      style R3 fill:#38d9a9,stroke:#0ca678,color:#000
      style R4 fill:#51cf66,stroke:#2f9e44,color:#000
  ```

  Each column above is one leaf region on the hours number line. Moving left to right = more study hours = higher predicted score. The step-jumps between columns (53.50 → 63.00 → 72.00 → 81.67) are where the tree's split thresholds (2.5, 5.5, 7.5) sit.

  > **Note:** Each leaf predicts the mean of all training examples in its region. With more depth, each region becomes smaller and predictions become more precise — but also more prone to overfitting. Pruning controls this by stopping splits whose variance reduction is too small to justify the added complexity.

</details>

---

## Regression Tree (n features)

A Regression Tree with multiple features searches over all features and all thresholds at every node, choosing the single (feature, threshold) pair that reduces MSE the most. Each internal node may test a different feature, allowing the tree to partition a multi-dimensional space into rectangular regions, each with its own constant predicted value.

**Splitting rule:**

```
for each feature f and threshold t:
    compute VR(f, t)

choose (f*, t*) = argmax VR(f, t)

split:
    left  = { yᵢ | xᵢ[f*] <= t* }
    right = { yᵢ | xᵢ[f*] >  t* }
```

Also written as:

```
Variance Reduction:
  VR(S, f, t) = MSE(S) - |Sₗ|/|S| * MSE(Sₗ) - |Sᵣ|/|S| * MSE(Sᵣ)
```

Where `S` is the current node's sample set, `Sₗ` and `Sᵣ` are the left and right subsets after the split, and `MSE` is the mean squared error of the target values in each set.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Exam Score Based on Study Hours and Sleep Hours</summary>
  <br/>

  Imagine predicting a student's exam score using two features: hours studied (x1) and hours of sleep the night before (x2).

  **Dataset:**

  | Hours Studied (x1) | Hours Sleep (x2) | Exam Score (y) |
  |--------------------|------------------|----------------|
  | 2                  | 4                | 48             |
  | 3                  | 5                | 53             |
  | 4                  | 4                | 57             |
  | 5                  | 6                | 63             |
  | 6                  | 7                | 68             |
  | 7                  | 6                | 72             |
  | 8                  | 8                | 76             |
  | 9                  | 7                | 80             |
  | 10                 | 9                | 85             |

  **Step 1 — Measure root node variance:**

  Root holds all 9 examples.

  ```
  ȳ(root) = (48+53+57+63+68+72+76+80+85) / 9 = 602 / 9 = 66.89
  MSE(root) = 141.43
  ```

  **Step 2 — Search all (feature, threshold) pairs:**

  The algorithm evaluates every candidate split across both features:

  | Feature   | Threshold | Left mean | Right mean | VR       |
  |-----------|-----------|-----------|------------|----------|
  | hours     | <= 2.5    | 48.00     | 69.25      | 44.5988  |
  | hours     | <= 3.5    | 50.50     | 71.57      | 76.7416  |
  | hours     | <= 4.5    | 52.67     | 74.00      | 101.1358 |
  | **hours** | **<= 5.5**| **55.25** | **76.20**  |**108.3710**|
  | hours     | <= 6.5    | 57.80     | 78.25      | 103.2599 |
  | hours     | <= 7.5    | 60.17     | 80.33      | 90.3765  |
  | hours     | <= 8.5    | 62.43     | 82.50      | 69.6305  |
  | sleep     | <= 4.5    | 52.50     | 71.00      | 59.1543  |
  | sleep     | <= 5.5    | 52.67     | 74.00      | 101.1358 |
  | sleep     | <= 6.5    | 58.60     | 77.25      | 85.8821  |
  | sleep     | <= 7.5    | 63.00     | 80.50      | 52.9321  |

  **Step 3 — Select the best split:**

  The pair `(hours, <= 5.5)` achieves the highest VR of **108.3710**.

  ```
  Left  child (hours <= 5.5, n=4): {48,53,57,63}  ȳₗ = 55.25  MSE = 30.19
  Right child (hours >  5.5, n=5): {68,72,76,80,85}  ȳᵣ = 76.20  MSE = 35.36
  ```

  **Step 4 — Grow one more level (depth = 2):**

  Best second-level splits:

  - Left child  → `hours <= 3.5` (VR = 22.5625): LL={48,53} ȳ=50.50 / LR={57,63} ȳ=60.00
  - Right child → `hours <= 8.5` (VR = 26.4600): RL={68,72,76} ȳ=72.00 / RR={80,85} ȳ=82.50

  **Step 5 — The Fitted Tree (depth = 2):**

 ```mermaid
  flowchart TD
      A["Root: hours <= 5.5? | n=9 | MSE=141.43 | mean=66.89 | hours chosen over sleep VR 108.37 vs 101.14"]
      B["hours <= 3.5? | n=4 | MSE=30.19 | mean=55.25"]
      C["hours <= 8.5? | n=5 | MSE=35.36 | mean=76.20"]
      D["Leaf: predict 50.50 | n=2 | values: 48, 53"]
      E["Leaf: predict 60.00 | n=2 | values: 57, 63"]
      F["Leaf: predict 72.00 | n=3 | values: 68, 72, 76"]
      G["Leaf: predict 82.50 | n=2 | values: 80, 85"]

      A -->|YES - hours <= 5.5| B
      A -->|NO  - hours > 5.5| C
      B -->|YES - hours <= 3.5| D
      B -->|NO  - hours > 3.5| E
      C -->|YES - hours <= 8.5| F
      C -->|NO  - hours > 8.5| G

      style A fill:#f0c040,stroke:#b8860b,color:#000
      style B fill:#f7b731,stroke:#e67e00,color:#000
      style C fill:#f7b731,stroke:#e67e00,color:#000
      style D fill:#74c0fc,stroke:#1971c2,color:#000
      style E fill:#74c0fc,stroke:#1971c2,color:#000
      style F fill:#74c0fc,stroke:#1971c2,color:#000
      style G fill:#74c0fc,stroke:#1971c2,color:#000
  ```

  **Step 6 — Prediction Examples:**

  | x1 (hours) | x2 (sleep) | Tree path                          | Predicted | Actual | Error  |
  |------------|------------|------------------------------------|-----------|--------|--------|
  | 3          | 5          | <=5.5 → YES → <=3.5 → YES         | 50.50     | 53     | −2.50  |
  | 5          | 6          | <=5.5 → YES → <=3.5 → NO          | 60.00     | 63     | −3.00  |
  | 7          | 7          | <=5.5 → NO  → <=8.5 → YES         | 72.00     | 72     |  0.00  |
  | 9          | 8          | <=5.5 → NO  → <=8.5 → NO          | 82.50     | 80     | +2.50  |

  **Interpreting the split:**

  The tree chose `hours` over `sleep` at the root because study hours produced a higher VR (108.37 vs 101.14 for the best sleep split). At the second level, both children again used `hours` — in this dataset study time is the dominant predictor. Sleep features would appear at deeper levels in a larger, noisier dataset where they contribute independent information.

  **2D feature-space partition:**

  The tree carves the (hours, sleep) plane into four rectangular regions with vertical cuts at `hours = 5.5` and `hours = 3.5` / `hours = 8.5`. Every point inside a region receives the same flat prediction regardless of where it falls within that rectangle.

  ```mermaid
  flowchart LR
      TL["hours <= 3.5 - any sleep | Leaf LL | predict 50.50 | values: 48, 53"]
      TR["3.5 < hours <= 5.5 - any sleep | Leaf LR | predict 60.00 | values: 57, 63"]
      BL["5.5 < hours <= 8.5 - any sleep | Leaf RL | predict 72.00 | values: 68, 72, 76"]
      BR["hours > 8.5 - any sleep | Leaf RR | predict 82.50 | values: 80, 85"]

      TL --> TR --> BL --> BR

      style TL fill:#dbe4ff,stroke:#4263eb,color:#000
      style TR fill:#74c0fc,stroke:#1971c2,color:#000
      style BL fill:#38d9a9,stroke:#0ca678,color:#000
      style BR fill:#51cf66,stroke:#2f9e44,color:#000
  ```

  Key observation: sleep hours did not appear in any split of this depth-2 tree — all four boundaries are vertical lines in the `hours` axis. This means the `sleep` axis is completely ignored at this depth, and all four regions span the full height of the sleep axis. A deeper tree or a noisier dataset would eventually introduce horizontal cuts driven by `sleep`.

  > **Note:** In practice, real datasets rarely yield such clean splits. The tree will grow multiple levels deep, and **pruning** (limiting max depth, minimum samples per leaf, or cost-complexity pruning) is applied to prevent overfitting.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

Regression Trees produce **step-function predictions** — within each leaf, every input receives the same constant prediction regardless of where it falls inside that region. This means they cannot extrapolate beyond the range of training data and struggle to capture smooth, continuous trends. They are also prone to **overfitting**: a deep tree memorizes the training set and produces poor generalization. Like Decision Trees, they are **unstable** — small changes in training data can completely restructure the tree. In such cases, alternatives include Linear Regression (for smooth global trends), Random Forests (which average many regression trees to reduce variance), Gradient Boosted Trees (which build trees sequentially to reduce bias), or Support Vector Regression (for kernel-based smooth fits).

---

## Error and the Cost Function

### Variance at Each Node

A Regression Tree does not minimize a global cost function the way gradient-based models do. Instead, at each node it greedily minimizes **local variance** — the spread of target values in the current subset. A node with all identical target values has zero variance and becomes a leaf; a node with widely scattered values has high variance and benefits from further splitting.

| Node contents                         | MSE     | Interpretation                      |
|---------------------------------------|---------|-------------------------------------|
| {68.5, 68.5, 68.5, 68.5} (n=4)       | 0.0000  | Zero variance — perfect leaf        |
| {52, 55, 60, 63, 66} (n=5)           | 26.1600 | Moderate variance — left child      |
| {70, 74, 78, 82, 85} (n=5)           | 28.9600 | Moderate variance — right child     |
| {52,55,60,63,66,70,74,78,82,85} (n=10)| 114.0500 | High variance — root before split  |

**Why use MSE instead of entropy or Gini?**

- **MSE** measures spread of continuous values — it equals zero only when all values in the node are identical, and grows as values become more scattered.
- **Entropy and Gini** are defined for class labels (probabilities), not for continuous targets — they cannot be applied to regression problems.
- **MAE (Mean Absolute Error)** is an alternative splitting criterion that is more robust to outliers but harder to optimize because it lacks a closed-form mean as the optimal leaf prediction.

---

### Objective: Maximize Variance Reduction at Each Split

The tree's goal at each node is to find the (feature, threshold) pair that reduces target variance the most — measured by **Variance Reduction (VR)**, which is equivalent to the reduction in MSE weighted by node sizes.

---

### Variance Reduction (VR)

- **Definition:** The reduction in MSE achieved by splitting node S into left and right children.
- **Formula:**

$$VR(S, f, t) = MSE(S) - \frac{|S_l|}{|S|} \cdot MSE(S_l) - \frac{|S_r|}{|S|} \cdot MSE(S_r)$$

- **Interpretation:** VR = 0 means the split provided no useful grouping of target values; VR = MSE(S) means both children have zero variance — all training targets in each child are identical.

**Example — root split at hours <= 5.5 (10 samples):**

```
MSE(root)  = 114.05  ({52,55,60,63,66,70,74,78,82,85}, ȳ=68.50)
MSE(left)  =  26.16  ({52,55,60,63,66}, ȳ=59.20)
MSE(right) =  28.96  ({70,74,78,82,85}, ȳ=77.80)

VR = 114.05 - (5/10)*26.16 - (5/10)*28.96
   = 114.05 - 13.08 - 14.48
   = 86.49
```

This split removes 86.49 / 114.05 = **75.8%** of the root variance — a strong split.

---

### Cost Function: Mean Squared Error (MSE)

MSE is both the splitting criterion and the training loss of a Regression Tree. It measures the average squared deviation of target values from their node mean.

**Formula:**

$$MSE(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

Where:
- `|S|` — number of samples in node S
- `yᵢ` — individual target value
- `ȳ` — mean of all target values in S

**Example Calculation (left child, n=5):**

$$MSE = \frac{(52-59.2)^2 + (55-59.2)^2 + (60-59.2)^2 + (63-59.2)^2 + (66-59.2)^2}{5}$$
$$= \frac{51.84 + 17.64 + 0.64 + 14.44 + 46.24}{5} = \frac{130.80}{5} = 26.16$$

To express error in the original units, take the square root (RMSE):

$$RMSE = \sqrt{26.16} \approx 5.11$$

On average, predictions in the left leaf are off by about **5.11 score points**.

---

### Alternative Notation (Leaf Prediction)

Each leaf predicts the **mean** of all training target values that fell into it during training. This is the value that minimizes MSE within the leaf:

$$\hat{y}_{\text{leaf}} = \bar{y}_{\text{leaf}} = \frac{1}{|S_{\text{leaf}}|} \sum_{i \in S_{\text{leaf}}} y_i$$

**Example (depth-2 tree, all leaves):**

| Leaf region              | Training values       | Predicted (mean) | MSE   |
|--------------------------|-----------------------|------------------|-------|
| hours <= 2.5             | {52, 55}              | 53.50            | 2.25  |
| 2.5 < hours <= 5.5       | {60, 63, 66}          | 63.00            | 6.00  |
| 5.5 < hours <= 7.5       | {70, 74}              | 72.00            | 4.00  |
| hours > 7.5              | {78, 82, 85}          | 81.67            | 8.22  |

A new input at `hours = 4` falls into region `2.5 < hours <= 5.5` and receives the prediction **63.00** regardless of where it sits within that interval — this is the step-function nature of Regression Trees.

---

## How Do We Find the Best Split?

At each node the tree performs an **exhaustive greedy search**: for every feature, sort the training examples by that feature's values and try every midpoint between adjacent values as a candidate threshold. Compute the VR for each, and pick the (feature, threshold) pair with the highest VR. This is repeated recursively for each child node until a stopping criterion is met (zero variance in a node, maximum depth reached, or too few samples to split).

---

## Recursive Splitting and Pruning

Regression Trees are built top-down by **recursive binary splitting** and controlled bottom-up by **pruning**.

**Recursive splitting process:**

```
build_tree(node, samples):
    if stopping_criterion_met:
        node.prediction = mean(target values in samples)
        return

    (f*, t*) = best_split(samples)
    node.feature    = f*
    node.threshold  = t*
    left, right     = split(samples, f*, t*)
    node.left       = build_tree(new_node, left)
    node.right      = build_tree(new_node, right)
```

**Stopping criteria** (any one triggers a leaf):
- Node variance is zero (MSE = 0)
- Maximum tree depth reached
- Fewer than `min_samples_split` examples remain
- VR of the best split is below a minimum threshold

**Pruning — Cost-Complexity Pruning (CCP):**

After the full tree is grown, CCP removes subtrees that do not justify their complexity. For each internal node, it computes:

$$\alpha = \frac{MSE(\text{node}) - MSE(\text{subtree})}{\text{leaves}(\text{subtree}) - 1}$$

Where `MSE(node)` is the error if the subtree is collapsed to a single leaf, `MSE(subtree)` is the weighted average MSE of all leaves in the subtree, and `leaves` is the number of leaf nodes. A subtree with a small `α` is pruned first — it costs very little prediction accuracy to collapse it into a leaf.

**Example:**

```
MSE(parent node as single leaf) = 14.0   (if we prune this subtree to a leaf)
MSE(subtree, weighted average)  =  6.0   (average MSE across 3 leaves)
leaves                          =  3

α = (14.0 - 6.0) / (3 - 1) = 8.0 / 2 = 4.0
```

If the chosen regularization strength exceeds α = 4.0, this subtree is pruned.

- A **shallow tree** (small max depth) underfits — too few splits to capture the data's structure, large residuals in every leaf.
- A **deep tree** (no pruning) overfits — each leaf contains very few samples, predicts training data perfectly, and fails on new inputs.

---

## Summary of Key Formulas

| Concept                        | Formula                                                                              |
|--------------------------------|--------------------------------------------------------------------------------------|
| Node MSE                       | MSE(S) = (1/\|S\|) * Σ (yᵢ − ȳ)²                                                    |
| Variance Reduction             | VR = MSE(S) − \|Sₗ\|/\|S\| * MSE(Sₗ) − \|Sᵣ\|/\|S\| * MSE(Sᵣ)                      |
| Leaf prediction                | ȳ\_leaf = mean(yᵢ for all i in leaf)                                                 |
| RMSE (leaf error in orig. units)| RMSE = sqrt(MSE)                                                                    |
| Optimal leaf value             | argmin\_c Σ (yᵢ − c)² = ȳ (the mean minimizes MSE)                                  |
| CCP pruning alpha              | α = (MSE(node) − MSE(subtree)) / (leaves(subtree) − 1)                              |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- Regression Trees — Scikit-learn *(coming soon)*

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">


<div align="center">
  <a href="https://www.youtube.com/watch?v=UhY5vPfQIrA" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/UhY5vPfQIrA/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=g9c66TUylZ4" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/g9c66TUylZ4/hqdefault.jpg"/>
  </a>
</div>