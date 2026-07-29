# Decision Trees <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

Decision Tree (1 feature) | Decision Tree (n features) | Splitting Criteria | Pruning

---

## What is a Decision Tree? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

A Decision Tree is a **supervised learning algorithm** used for both **classification** (e.g., pass/fail, spam/not spam) and **regression** (e.g., predicting a price or score) by learning a sequence of **if-then-else rules** from the data.

At its core, it recursively partitions the feature space into regions by asking a series of binary questions at each node. Each internal node tests a feature against a threshold; each branch follows the outcome of that test; each leaf node holds the final prediction. The result is a tree-shaped model that is naturally interpretable — you can follow any prediction path from root to leaf and read the exact reasoning in plain English.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to build a tree that correctly classifies or predicts training examples while remaining as simple as possible. Three concepts are central:

- **Node** — a decision point that tests one feature against a threshold (`x <= t`). The root node is the first split; internal nodes are subsequent splits.
- **Branch** — the outcome of a test (left branch = condition true; right branch = condition false).
- **Leaf** — a terminal node that holds the final prediction: the majority class (classification) or the mean target value (regression).

At every node, the algorithm searches over all features and all thresholds to find the split that **reduces impurity the most**. Impurity measures how mixed the class labels are in a node — a pure node (all one class) has zero impurity.

**Splitting criteria:**

```
Entropy  H(S) = - Σ pᵢ * log₂(pᵢ)
Gini     G(S) = 1 - Σ pᵢ²
```

Where `pᵢ` is the proportion of class `i` in node `S`. Both reach 0 for a pure node and are maximized when all classes are equally represented.

---

## Decision Tree (1 feature)

A Decision Tree with a single feature partitions the number line into intervals by choosing one or more threshold values. Each threshold creates a binary question: "Is x ≤ t?" The algorithm picks the threshold that yields the greatest reduction in impurity.

**Splitting rule:**

```
if x <= threshold  →  go left  (one prediction)
if x >  threshold  →  go right (another prediction)
```

Also written as:

```
split(x, t):
    left  = { xᵢ | xᵢ <= t }
    right = { xᵢ | xᵢ >  t }
    choose t that maximizes Information Gain (or Gini Gain)
```

Where:

- `x` — the single input feature
- `t` — the threshold value being tested at the node
- `left`, `right` — the subsets of training examples that go into each child node
- `Information Gain` — the reduction in entropy achieved by the split
- `Gini Gain` — the reduction in Gini impurity achieved by the split

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Classifying Exam Pass/Fail Based on Study Hours</summary>
  <br/>

  Imagine classifying whether a student passes or fails based on how many hours they studied.

  **Dataset (Hours Studied vs Result):**

  | Hours Studied (x) | Result (y) |
  |-------------------|------------|
  | 1                 | Fail       |
  | 2                 | Fail       |
  | 3                 | Fail       |
  | 4                 | Fail       |
  | 5                 | Pass       |
  | 6                 | Pass       |
  | 7                 | Pass       |
  | 8                 | Pass       |
  | 9                 | Pass       |
  | 10                | Pass       |

  **Step 1 — Measure root node impurity:**

  The root node holds all 10 examples: 4 Fail and 6 Pass.

  ```
  p(Fail) = 4/10 = 0.4,   p(Pass) = 6/10 = 0.6

  Entropy H = -0.4*log₂(0.4) - 0.6*log₂(0.6)
            = -0.4*(-1.3219) - 0.6*(-0.7370)
            = 0.5288 + 0.4422
            = 0.9710

  Gini G = 1 - (0.4)² - (0.6)² = 1 - 0.16 - 0.36 = 0.4800
  ```

  **Step 2 — Evaluate all candidate splits:**

  The algorithm tries every midpoint between consecutive feature values as a threshold and computes the Information Gain (IG) for each:

  | Threshold     | Left node          | Right node         | IG     |
  |---------------|--------------------|--------------------|--------|
  | hours <= 1.5  | {1F}               | {3F, 6P}           | 0.1445 |
  | hours <= 2.5  | {2F}               | {2F, 6P}           | 0.3219 |
  | hours <= 3.5  | {3F}               | {1F, 6P}           | 0.5568 |
  | **hours <= 4.5**  | **{4F}**       | **{6P}**           | **0.9710** |
  | hours <= 5.5  | {4F, 1P}           | {5P}               | 0.6100 |
  | hours <= 6.5  | {4F, 2P}           | {4P}               | 0.4200 |
  | hours <= 7.5  | {4F, 3P}           | {3P}               | 0.2813 |
  | hours <= 8.5  | {4F, 4P}           | {2P}               | 0.1710 |
  | hours <= 9.5  | {4F, 5P}           | {1P}               | 0.0790 |

  **Step 3 — Select the best split:**

  The threshold `hours <= 4.5` achieves the highest IG of **0.9710** — it perfectly separates all Fail examples (left) from all Pass examples (right).

  ```
  Left  child (hours <= 4.5): {4 Fail, 0 Pass}  →  H = 0.0  (pure)
  Right child (hours >  4.5): {0 Fail, 6 Pass}  →  H = 0.0  (pure)
  ```

  Both children are pure leaves — no further splitting is needed.

  **Step 4 — The Fitted Tree:**

  ```mermaid
  flowchart TD
      A["🌿 Root node\nhours <= 4.5?\nn=10 | H=0.9710 | G=0.4800"]
      A -->|YES| B["🍂 Leaf\npredict: Fail\nn=4 | H=0.0 | G=0.0\n{4 Fail, 0 Pass}"]
      A -->|NO| C["🍂 Leaf\npredict: Pass\nn=6 | H=0.0 | G=0.0\n{0 Fail, 6 Pass}"]

      style A fill:#f0c040,stroke:#b8860b,color:#000
      style B fill:#ff6b6b,stroke:#c0392b,color:#fff
      style C fill:#51cf66,stroke:#2f9e44,color:#fff
  ```

  **Prediction Example:**

  What class does a student with 3 hours of study get?

  ```
  hours = 3 <= 4.5  →  go left  →  predict: Fail
  ```

  And a student with 7 hours?

  ```
  hours = 7 > 4.5  →  go right  →  predict: Pass
  ```

  **Visual Analogy:**

  Imagine plotting students on a number line by hours studied, colored by pass/fail. The decision tree draws a vertical dividing line at 4.5. Every student to the left is predicted Fail; every student to the right is predicted Pass. The position of that line is chosen to minimize the mixing of colors on either side.

  > **Note:** This tree reached perfect purity in one split because the data is cleanly separable. In practice, trees may need multiple levels of splits, and a maximum depth or minimum samples per leaf is set to prevent overfitting.

</details>

---

## Decision Tree (n features)

A Decision Tree with multiple features searches over all features and all thresholds at every node, choosing the single (feature, threshold) pair that reduces impurity the most. Each internal node may test a different feature, allowing the tree to carve out complex, axis-aligned regions in the feature space.

**Splitting rule:**

```
for each feature f and threshold t:
    compute IG(f, t)  or  GG(f, t)

choose (f*, t*) = argmax IG(f, t)

split:
    left  = { xᵢ | xᵢ[f*] <= t* }
    right = { xᵢ | xᵢ[f*] >  t* }
```

Also written as:

```
Information Gain:  IG(S, f, t) = H(S) - |Sₗ|/|S| * H(Sₗ) - |Sᵣ|/|S| * H(Sᵣ)
Gini Gain:         GG(S, f, t) = G(S) - |Sₗ|/|S| * G(Sₗ) - |Sᵣ|/|S| * G(Sᵣ)
```

Where `S` is the current node's sample set, `Sₗ` and `Sᵣ` are the left and right subsets after the split, and `H` / `G` are entropy and Gini impurity respectively.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Classifying Exam Pass/Fail Based on Study Hours and Sleep Hours</summary>
  <br/>

  Imagine classifying students as Pass or Fail using two features: hours studied (x1) and hours of sleep the night before (x2).

  **Dataset:**

  | Hours Studied (x1) | Hours Sleep (x2) | Result (y) |
  |--------------------|------------------|------------|
  | 2                  | 4                | Fail       |
  | 3                  | 5                | Fail       |
  | 4                  | 4                | Fail       |
  | 5                  | 6                | Fail       |
  | 6                  | 7                | Pass       |
  | 7                  | 6                | Pass       |
  | 8                  | 8                | Pass       |
  | 9                  | 7                | Pass       |
  | 10                 | 9                | Pass       |

  **Step 1 — Measure root node impurity:**

  Root holds all 9 examples: 4 Fail and 5 Pass.

  ```
  p(Fail) = 4/9,  p(Pass) = 5/9
  H(root) = -(4/9)*log₂(4/9) - (5/9)*log₂(5/9) = 0.9911
  G(root) = 1 - (4/9)² - (5/9)² = 0.4938
  ```

  **Step 2 — Search all (feature, threshold) pairs:**

  The algorithm evaluates every candidate split across both features:

  | Feature   | Threshold | Left            | Right        | IG     |
  |-----------|-----------|-----------------|--------------|--------|
  | hours     | <= 2.5    | {1F}            | {3F, 5P}     | 0.1427 |
  | hours     | <= 3.5    | {2F}            | {2F, 5P}     | 0.3198 |
  | hours     | <= 4.5    | {3F}            | {1F, 5P}     | 0.5577 |
  | **hours** | **<= 5.5**| **{4F}**        | **{5P}**     | **0.9911** |
  | hours     | <= 6.5    | {4F, 1P}        | {4P}         | 0.5900 |
  | sleep     | <= 4.5    | {2F}            | {2F, 5P}     | 0.3198 |
  | sleep     | <= 5.5    | {3F}            | {1F, 5P}     | 0.5577 |
  | sleep     | <= 6.5    | {4F, 1P}        | {4P}         | 0.5900 |

  **Step 3 — Select the best split:**

  The pair `(hours, <= 5.5)` achieves the highest IG of **0.9911** — it perfectly separates Fail from Pass.

  ```
  Left  child (hours <= 5.5): {4 Fail, 0 Pass}  →  H = 0.0  (pure, predict Fail)
  Right child (hours >  5.5): {0 Fail, 5 Pass}  →  H = 0.0  (pure, predict Pass)
  ```

  Both children are pure — the tree stops here.

  **Step 4 — The Fitted Tree:**

  ```mermaid
  flowchart TD
      A["🌿 Root node\nhours <= 5.5?\nn=9 | H=0.9911 | G=0.4938\nFeatures considered: hours ✓  sleep ✗"]
      A -->|YES| B["🍂 Leaf\npredict: Fail\nn=4 | H=0.0 | G=0.0\n{4 Fail, 0 Pass}"]
      A -->|NO| C["🍂 Leaf\npredict: Pass\nn=5 | H=0.0 | G=0.0\n{0 Fail, 5 Pass}"]

      D["💤 sleep features\nIG best = 0.5900\n❌ not chosen"]
      E["📚 hours feature\nIG best = 0.9911\n✅ chosen"]

      F["Candidate splits evaluated at root"]
      F --> D
      F --> E
      E --> A

      style A fill:#f0c040,stroke:#b8860b,color:#000
      style B fill:#ff6b6b,stroke:#c0392b,color:#fff
      style C fill:#51cf66,stroke:#2f9e44,color:#fff
      style D fill:#adb5bd,stroke:#6c757d,color:#000
      style E fill:#74c0fc,stroke:#1971c2,color:#000
      style F fill:#e9ecef,stroke:#adb5bd,color:#000
  ```

  **Step 5 — Prediction Examples:**

  | x1 (hours) | x2 (sleep) | Tree path               | Predicted | Actual |
  |------------|------------|-------------------------|-----------|--------|
  | 3          | 5          | 3 <= 5.5 → left         | Fail      | Fail   |
  | 7          | 7          | 7 > 5.5  → right        | Pass      | Pass   |
  | 5          | 7          | 5 <= 5.5 → left         | Fail      | Fail   |
  | 6          | 6          | 6 > 5.5  → right        | Pass      | Pass   |

  **Interpreting the split:**

  The tree chose `hours` over `sleep` at the root because study hours provided a perfect separation (IG = 0.9911) while sleep hours alone could not achieve the same purity at any threshold. Each internal node always picks the single most informative feature at that point — features that do not help are simply never used.

  > **Note:** In practice, real datasets rarely yield perfect splits. The tree will grow multiple levels deep, and **pruning** (limiting max depth, minimum samples per leaf, or cost-complexity pruning) is applied to prevent overfitting.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

Decision Trees are prone to **overfitting** — a deep tree can memorize every training example perfectly while failing badly on new data. They are also **unstable**: a small change in the training data can produce a completely different tree. The decision boundaries are always **axis-aligned** (each split tests only one feature), which makes them inefficient for diagonal or curved class boundaries. In such cases, alternatives include Random Forests (which average many trees to reduce variance), Gradient Boosted Trees (which build trees sequentially to reduce bias), or Support Vector Machines (for clean linear or kernel boundaries). Pruning techniques — setting a maximum depth, minimum samples per leaf, or using cost-complexity pruning (CCP) — are the standard way to control overfitting within a single tree.

---

## Error and the Cost Function

### Impurity Measures

A Decision Tree does not minimize a global cost function the way gradient-based models do. Instead, at each node it greedily minimizes **local impurity** — the degree of class mixing in the current subset. A node is impure when it contains examples from multiple classes; it is pure when all examples belong to one class.

| Node contents       | Entropy | Gini   | Interpretation            |
|---------------------|---------|--------|---------------------------|
| 5 Pass, 5 Fail (n=10) | 1.0000 | 0.5000 | Maximum impurity (50/50)  |
| 4 Fail, 6 Pass (n=10) | 0.9710 | 0.4800 | High impurity             |
| 1 Fail, 5 Pass (n=6)  | 0.6500 | 0.2778 | Moderate impurity         |
| 0 Fail, 6 Pass (n=6)  | 0.0000 | 0.0000 | Pure node — no error      |

**Why use entropy or Gini instead of accuracy?**

- **Entropy** penalizes uncertainty logarithmically — it drops steeply as a node becomes purer, making it sensitive to small purity improvements.
- **Gini** is faster to compute (no logarithm) and behaves similarly to entropy in practice. It reaches 0 for a pure node and 0.5 for an even two-class split.
- **Raw accuracy** is not used because it does not distinguish between "slightly impure" and "very impure" splits — it changes only in discrete jumps.

---

### Objective: Maximize Information Gain at Each Split

The tree's goal at each node is to find the (feature, threshold) pair that reduces impurity the most — measured by **Information Gain** (entropy-based) or **Gini Gain** (Gini-based).

---

### Information Gain (IG)

- **Definition:** The reduction in entropy achieved by splitting node S into left and right children.
- **Formula:**

$$IG(S, f, t) = H(S) - \frac{|S_l|}{|S|} H(S_l) - \frac{|S_r|}{|S|} H(S_r)$$

- **Interpretation:** IG = 0 means the split provided no useful information; IG = H(S) means the split made both children perfectly pure.

**Example — root split at hours <= 4.5 (10 samples):**

```
H(root) = 0.9710  (4 Fail, 6 Pass)
H(left) = 0.0     (4 Fail — pure)
H(right)= 0.0     (6 Pass — pure)

IG = 0.9710 - (4/10)*0.0 - (6/10)*0.0 = 0.9710
```

A gain equal to the root entropy means the split eliminated all impurity — the best possible outcome.

---

### Cost Function: Gini Impurity

Gini impurity is the default splitting criterion in many libraries (e.g., scikit-learn). It measures the probability that a randomly chosen sample would be incorrectly classified if labeled according to the class distribution in the node.

**Formula:**

$$G(S) = 1 - \sum_{i=1}^{k} p_i^2$$

Where:
- `k` — number of classes
- `pᵢ` — proportion of class `i` in node `S`

**Example Calculation (root node, 4 Fail and 6 Pass, n=10):**

$$G = 1 - \left(\frac{4}{10}\right)^2 - \left(\frac{6}{10}\right)^2 = 1 - 0.16 - 0.36 = 0.4800$$

**Gini Gain** after splitting at `hours <= 4.5`:

$$GG = 0.4800 - \frac{4}{10} \cdot 0.0 - \frac{6}{10} \cdot 0.0 = 0.4800$$

The full impurity was eliminated — both children are pure.

---

### Alternative Notation (Multi-class Gini)

For problems with more than two classes, Gini impurity generalizes directly:

$$G(S) = 1 - \sum_{i=1}^{k} p_i^2$$

**Example (node with 3 classes: A=2, B=3, C=1, n=6):**

$$G = 1 - \left(\frac{2}{6}\right)^2 - \left(\frac{3}{6}\right)^2 - \left(\frac{1}{6}\right)^2$$
$$= 1 - 0.1111 - 0.2500 - 0.0278 = 0.6111$$

**IG worked example (imperfect split):**

Suppose the parent has 5 Pass and 5 Fail (n=10, H=1.0). A candidate split gives:

- Left child: 4 Fail (n=4) → H = 0.0
- Right child: 5 Pass, 1 Fail (n=6) → H = 0.6500

$$IG = 1.0 - \frac{4}{10} \cdot 0.0 - \frac{6}{10} \cdot 0.6500 = 1.0 - 0.0 - 0.3900 = 0.6100$$

This split is good (IG = 0.6100) but not perfect — the right child still contains one Fail example, so the tree would continue splitting it at the next level.

---

## How Do We Find the Best Split?

At each node the tree performs an **exhaustive greedy search**: for every feature, sort the training examples by that feature's values and try every midpoint between adjacent values as a candidate threshold. Compute the IG or Gini Gain for each, and pick the (feature, threshold) pair with the highest gain. This is repeated recursively for each child node until a stopping criterion is met (pure leaves, maximum depth reached, or too few samples to split).

---

## Recursive Splitting and Pruning

Decision Trees are built top-down by **recursive binary splitting** and controlled bottom-up by **pruning**.

**Recursive splitting process:**

```
build_tree(node, samples):
    if stopping_criterion_met:
        node.label = majority_class(samples)
        return

    (f*, t*) = best_split(samples)
    node.feature    = f*
    node.threshold  = t*
    left, right     = split(samples, f*, t*)
    node.left       = build_tree(new_node, left)
    node.right      = build_tree(new_node, right)
```

**Stopping criteria** (any one triggers a leaf):
- Node is pure (H = 0 or G = 0)
- Maximum tree depth reached
- Fewer than `min_samples_split` examples remain
- IG of the best split is below a minimum threshold

**Pruning — Cost-Complexity Pruning (CCP):**

After the full tree is grown, CCP removes subtrees that do not justify their complexity. For each internal node, it computes:

$$\alpha = \frac{R(\text{node}) - R(\text{subtree})}{\text{leaves}(\text{subtree}) - 1}$$

Where `R` is the weighted misclassification rate and `leaves` is the number of leaf nodes in the subtree. A subtree with a small `α` is pruned first — it costs very little accuracy to collapse it into a leaf.

**Example:**

```
R(parent node)  = 0.30   (30% misclassification if we collapse to a leaf)
R(subtree)      = 0.10   (10% weighted misclassification across 3 leaves)
leaves          = 3

α = (0.30 - 0.10) / (3 - 1) = 0.20 / 2 = 0.10
```

If the chosen regularization strength exceeds α = 0.10, this subtree is pruned.

- A **shallow tree** (small max depth) underfits — too few splits to capture the data's structure.
- A **deep tree** (no pruning) overfits — memorizes training data and fails on new examples.

---

## Summary of Key Formulas

| Concept                      | Formula                                                                           |
|------------------------------|-----------------------------------------------------------------------------------|
| Entropy                      | H(S) = -Σ pᵢ * log₂(pᵢ)                                                          |
| Gini impurity                | G(S) = 1 - Σ pᵢ²                                                                 |
| Information Gain             | IG = H(S) - \|Sₗ\|/\|S\| * H(Sₗ) - \|Sᵣ\|/\|S\| * H(Sᵣ)                         |
| Gini Gain                    | GG = G(S) - \|Sₗ\|/\|S\| * G(Sₗ) - \|Sᵣ\|/\|S\| * G(Sᵣ)                         |
| Leaf prediction (class.)     | majority\_class(samples in leaf)                                                  |
| Leaf prediction (regression) | mean(target values in leaf)                                                       |
| CCP pruning alpha            | α = (R(node) − R(subtree)) / (leaves(subtree) − 1)                               |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- Decision Trees — Scikit-learn *(coming soon)*

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=_L39rN6gz7Y" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/_L39rN6gz7Y/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=ZVR2Way4nwQ" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/ZVR2Way4nwQ/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=q90UDEgYqeI" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/q90UDEgYqeI/hqdefault.jpg"/>
  </a>
</div>