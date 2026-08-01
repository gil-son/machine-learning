# Random Forest <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

Random Forest Classification (1 feature) | Random Forest Classification (n features) | Regression | OOB Error

---

## What is a Random Forest? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

A Random Forest is a **supervised ensemble learning algorithm** used for both **classification** (e.g., pass/fail, spam/not spam) and **regression** (e.g., predicting a score or price) by building a large number of **Decision Trees** and combining their predictions.

At its core, it addresses the main weakness of a single Decision Tree — instability and overfitting — by introducing two sources of randomness: **bootstrap sampling** (each tree trains on a different random subset of the data) and **random feature selection** (each split considers only a random subset of features). The result is a collection of diverse, decorrelated trees whose averaged or majority-voted prediction is far more robust than any single tree.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to build an ensemble of Decision Trees, each trained on a different bootstrapped version of the data and using a random subset of features at each split, then combine their predictions into a single robust output. Four concepts are central:

- **Bootstrap sampling** — each tree trains on `n` samples drawn with replacement from the training set. On average, ~63.2% of original samples appear in each bootstrap; the remaining ~36.8% form the **Out-Of-Bag (OOB)** set used for free validation.
- **Random feature selection** — at each split node, only `m` randomly chosen features are considered (typically `m = sqrt(p)` for classification, `m = p/3` for regression, where `p` is the total number of features). This decorrelates the trees.
- **Aggregation** — classification uses **majority vote** across all trees; regression uses the **mean** of all tree predictions.
- **Feature importance** — measured by the average reduction in impurity (Gini or MSE) that each feature contributes across all splits in all trees.

**Prediction rules:**

```
Classification:  y_hat = majority_vote( tree_1(x), tree_2(x), ..., tree_T(x) )
Regression:      y_hat = (1/T) * sum( tree_t(x)  for t in 1..T )
```

Where `T` is the total number of trees in the forest.

---

## Random Forest Classification (1 feature)

With a single feature, each tree in the forest is a stump (or shallow tree) trained on a bootstrap sample of the data. Because the samples differ, the trees learn slightly different split thresholds, and the majority vote over all of them is more stable than any individual prediction.

**Prediction rule:**

```
y_hat = majority_vote( tree_1(x), ..., tree_T(x) )
```

Also written as:

```
y_hat = argmax_c  sum( 1[tree_t(x) == c]  for t in 1..T )
```

Where:

- `y_hat` — predicted class label
- `T` — number of trees in the forest
- `tree_t(x)` — prediction of the t-th tree for input x
- `majority_vote` — the class predicted by more than half the trees
- `1[·]` — indicator function: 1 if condition is true, 0 otherwise

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Classifying Exam Pass/Fail Based on Study Hours</summary>
  <br/>

  Imagine classifying whether a student passes or fails an exam based on hours studied, using a forest of 3 trees.

  **Original Dataset (Hours Studied vs Result):**

  | Index | Hours (x) | Label (y) |
  |-------|-----------|-----------|
  | 0     | 1         | Fail      |
  | 1     | 2         | Fail      |
  | 2     | 3         | Fail      |
  | 3     | 4         | Fail      |
  | 4     | 5         | Pass      |
  | 5     | 6         | Pass      |
  | 6     | 7         | Pass      |
  | 7     | 8         | Pass      |
  | 8     | 9         | Pass      |
  | 9     | 10        | Pass      |

  **Step 1 — Bootstrap: draw 3 different training sets:**

  Each tree trains on 10 samples drawn with replacement from the original 10. Different indices can appear multiple times; some are left out (OOB).

  | Tree | Bootstrap indices drawn       | Hours in bootstrap             | OOB indices | OOB hours |
  |------|-------------------------------|--------------------------------|-------------|-----------|
  | 1    | 4,9,6,3,3,7,7,9,7,8          | 5,10,7,4,4,8,8,10,8,9 (Pass heavy) | 0,1,2,5 | 1,2,3,6 |
  | 2    | 2,0,0,6,2,4,9,3,4,2          | 3,1,1,7,3,5,10,4,5,3 (Fail heavy) | 1,5,7,8 | 2,6,8,9 |
  | 3    | 9,8,4,0,0,8,3,2,1,8          | 10,9,5,1,1,9,4,3,2,9 (mixed)      | 5,6,7   | 6,7,8   |

  **Step 2 — Train one stump per tree:**

  Each stump searches for the best threshold on its bootstrap data:

  ```mermaid
  flowchart TD
      T1["Tree 1 bootstrap - hours: 4,4,5,7,8,8,8,9,10,10"]
      T1L["Leaf: predict Fail - hours <= 4.5 - 2 Fail"]
      T1R["Leaf: predict Pass - hours > 4.5 - 8 Pass"]
      T1 -->|YES - hours <= 4.5| T1L
      T1 -->|NO  - hours > 4.5| T1R

      T2["Tree 2 bootstrap - hours: 1,1,3,3,3,4,5,5,7,10"]
      T2L["Leaf: predict Fail - hours <= 4.5 - 7 Fail, 0 Pass"]
      T2R["Leaf: predict Pass - hours > 4.5 - 0 Fail, 3 Pass"]
      T2 -->|YES - hours <= 4.5| T2L
      T2 -->|NO  - hours > 4.5| T2R

      T3["Tree 3 bootstrap - hours: 1,1,2,3,4,5,9,9,9,10"]
      T3L["Leaf: predict Fail - hours <= 5.5 - 5 Fail, 1 Pass"]
      T3R["Leaf: predict Pass - hours > 5.5 - 0 Fail, 4 Pass"]
      T3 -->|YES - hours <= 5.5| T3L
      T3 -->|NO  - hours > 5.5| T3R

      style T1 fill:#f0c040,stroke:#b8860b,color:#000
      style T1L fill:#ff6b6b,stroke:#c0392b,color:#fff
      style T1R fill:#51cf66,stroke:#2f9e44,color:#fff
      style T2 fill:#f0c040,stroke:#b8860b,color:#000
      style T2L fill:#ff6b6b,stroke:#c0392b,color:#fff
      style T2R fill:#51cf66,stroke:#2f9e44,color:#fff
      style T3 fill:#f0c040,stroke:#b8860b,color:#000
      style T3L fill:#ff6b6b,stroke:#c0392b,color:#fff
      style T3R fill:#51cf66,stroke:#2f9e44,color:#fff
  ```

  **Step 3 — Majority vote for a new query:**

  What class does a student with **6 hours** of study get predicted?

  | Tree | Split threshold | 6 hours goes to | Prediction |
  |------|-----------------|-----------------|------------|
  | 1    | hours <= 4.5    | right (6 > 4.5) | Pass       |
  | 2    | hours <= 4.5    | right (6 > 4.5) | Pass       |
  | 3    | hours <= 5.5    | right (6 > 5.5) | Pass       |

  ```
  Vote: Pass=3, Fail=0  →  final prediction: Pass
  ```

  **Step 4 — OOB validation (free, no separate test set needed):**

  Each sample was OOB (not used in training) for at least one tree. We predict each OOB sample using only the trees that did not train on it:

  | Sample | Hours | True label | OOB predictions          | Final OOB vote | Correct? |
  |--------|-------|------------|--------------------------|----------------|----------|
  | 0      | 1     | Fail       | Tree 1: Fail             | Fail           | Yes      |
  | 1      | 2     | Fail       | Tree 1: Fail, Tree 2: Fail | Fail         | Yes      |
  | 2      | 3     | Fail       | Tree 1: Fail             | Fail           | Yes      |
  | 5      | 6     | Pass       | Tree 1: Pass, Tree 2: Pass, Tree 3: Pass | Pass | Yes |
  | 6      | 7     | Pass       | Tree 3: Pass             | Pass           | Yes      |
  | 7      | 8     | Pass       | Tree 2: Pass, Tree 3: Pass | Pass         | Yes      |
  | 8      | 9     | Pass       | Tree 2: Pass             | Pass           | Yes      |

  ```
  OOB accuracy = 7/7 = 100%   OOB error = 0%
  ```

  **Visual Analogy:**

  Imagine asking three different teachers (each having studied a different subset of students) to predict whether a new student will pass. Each teacher gives their answer independently. You go with the majority answer. Even if one teacher's sample happened to be unrepresentative, the others compensate — the crowd is more reliable than any individual.

  > **Note:** In practice a forest uses hundreds of trees (typically 100–500). More trees reduce variance further but increase training time. The OOB error is a reliable estimate of generalization error and often eliminates the need for a separate validation set.

</details>

---

## Random Forest Classification (n features)

With multiple features, each split in each tree is restricted to a **random subset of `m` features** chosen at that node. This prevents any single dominant feature from appearing at the root of every tree, forcing the trees to be diverse and decorrelated. The majority vote across all trees is then taken as the final prediction.

**Prediction rule:**

```
y_hat = majority_vote( tree_1(x), ..., tree_T(x) )
```

Also written as:

```
At each split node in each tree:
    F_candidate = random_sample(all_features, size=m)
    (f*, t*) = argmax IG(f, t)  for f in F_candidate

y_hat = argmax_c  sum( 1[tree_t(x) == c]  for t in 1..T )
```

Where:

- `m` — number of features randomly sampled at each split (hyperparameter; default `sqrt(p)`)
- `F_candidate` — the random subset of features considered at one node
- `IG(f, t)` — Information Gain of splitting on feature `f` at threshold `t`
- All other terms as in the 1-feature case

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Classifying Exam Pass/Fail Based on Study Hours and Sleep Hours</summary>
  <br/>

  Imagine classifying students as Pass or Fail using two features: hours studied (x1) and hours of sleep (x2), with a forest of 3 trees and `m = 1` feature sampled per split.

  **Dataset:**

  | Hours Studied (x1) | Hours Sleep (x2) | Label (y) |
  |--------------------|------------------|-----------|
  | 2                  | 4                | Fail      |
  | 3                  | 5                | Fail      |
  | 4                  | 4                | Fail      |
  | 5                  | 6                | Fail      |
  | 6                  | 7                | Pass      |
  | 7                  | 6                | Pass      |
  | 8                  | 8                | Pass      |
  | 9                  | 7                | Pass      |
  | 10                 | 9                | Pass      |

  **Step 1 — Identify the features and their importance:**

  Before building the forest, we can measure how useful each feature is by computing the best Gini Gain available from each one on the full dataset:

  - `x1` (hours studied): best Gini Gain = **0.4938** at threshold hours <= 5.5
  - `x2` (hours sleep):   best Gini Gain = **0.3160** at threshold sleep <= 6.5

  Study hours is the stronger feature — but both contribute. By randomly restricting each split to one feature at a time, some trees will be forced to use sleep, making the forest's collective knowledge broader.

  **Step 2 — Each tree trains on a bootstrap sample with random feature selection:**

  At every split node, only `m = 1` randomly drawn feature is evaluated. Different trees end up using different features at their roots:

  ```mermaid
  flowchart TD
      A["Tree 1 - root split on hours - best GG=0.4938"]
      A -->|YES - hours <= 5.5| B["Leaf: predict Fail"]
      A -->|NO  - hours > 5.5| C["Leaf: predict Pass"]

      D["Tree 2 - root split on sleep - forced by random selection - GG=0.3160"]
      D -->|YES - sleep <= 6.5| E["Leaf: predict Fail"]
      D -->|NO  - sleep > 6.5| F["Leaf: predict Pass"]

      G["Tree 3 - root split on hours - best GG=0.4938"]
      G -->|YES - hours <= 5.5| H["Leaf: predict Fail"]
      G -->|NO  - hours > 5.5| I["Leaf: predict Pass"]

      style A fill:#f0c040,stroke:#b8860b,color:#000
      style B fill:#ff6b6b,stroke:#c0392b,color:#fff
      style C fill:#51cf66,stroke:#2f9e44,color:#fff
      style D fill:#f0c040,stroke:#b8860b,color:#000
      style E fill:#ff6b6b,stroke:#c0392b,color:#fff
      style F fill:#51cf66,stroke:#2f9e44,color:#fff
      style G fill:#f0c040,stroke:#b8860b,color:#000
      style H fill:#ff6b6b,stroke:#c0392b,color:#fff
      style I fill:#51cf66,stroke:#2f9e44,color:#fff
  ```

  **Step 3 — Prediction Example:**

  What class does a student with **6 hours of study and 7 hours of sleep** get predicted?

  | Tree | Feature used at root | Query path               | Prediction |
  |------|----------------------|--------------------------|------------|
  | 1    | hours <= 5.5         | 6 > 5.5 → right          | Pass       |
  | 2    | sleep <= 6.5         | 7 > 6.5 → right          | Pass       |
  | 3    | hours <= 5.5         | 6 > 5.5 → right          | Pass       |

  ```
  Vote: Pass=3, Fail=0  →  final prediction: Pass
  ```

  **Interpreting feature importance:**

  After training the full forest, feature importance is computed as the average Gini Gain reduction that each feature contributes across all splits in all trees. Features that appear at the root of many trees and produce large impurity reductions receive high importance scores.

  | Feature        | Avg Gini Gain | Normalized Importance |
  |----------------|---------------|-----------------------|
  | hours (x1)     | 0.4938        | 61%                   |
  | sleep (x2)     | 0.3160        | 39%                   |

  > **Note:** Random feature selection is the key difference from Bagging. It forces diversity even when one feature is strongly dominant — preventing all trees from learning the same boundary and ensuring the ensemble captures multiple perspectives on the data.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

Random Forests are hard to interpret — unlike a single Decision Tree, there is no simple path from root to leaf to explain any individual prediction. They are also **memory-intensive**: storing hundreds of full-depth trees requires substantial RAM. Training time grows linearly with the number of trees. For **very high-dimensional sparse data** (e.g., text), linear models often outperform them. When prediction errors are correlated across trees (e.g., all trees share the same dominant feature even with subsampling), the ensemble gains less than expected. In those cases, alternatives include XGBoost or other Gradient Boosted Trees (which reduce bias by building trees sequentially rather than in parallel), or regularized linear models for high-dimensional settings.

---

## Error and the Cost Function

### Individual Tree Errors

A Random Forest inherits the splitting criteria of its constituent trees — **Gini impurity or Information Gain** for classification, **MSE** for regression — applied independently on each bootstrap sample. The ensemble error is always lower than the average individual tree error because averaging (or voting) cancels out the independent noise each tree makes.

| Query (hours) | True label | Tree 1 pred | Tree 2 pred | Tree 3 pred | RF vote | Correct? |
|---------------|------------|-------------|-------------|-------------|---------|----------|
| 6             | Pass       | Pass        | Pass        | Pass        | Pass    | Yes      |
| 4             | Fail       | Fail        | Fail        | Fail        | Fail    | Yes      |
| 5             | Pass       | Pass        | Fail        | Fail        | Fail    | No       |

The third row shows that individual trees can disagree near the decision boundary. The majority vote dampens but does not eliminate boundary errors — reducing the variance of the prediction without increasing bias.

---

### Objective: Minimize Ensemble Error Through Diversity

The forest's goal is not to minimize one global cost function, but to build a collection of trees that are each **individually accurate** and **mutually decorrelated**. Error theory (Breiman, 2001) shows:

```
Ensemble error <= mean_tree_error * (1 - mean_correlation)
```

The lower the correlation between trees (achieved by bootstrap sampling and random feature selection), the greater the error reduction relative to a single tree.

---

### OOB Error — Out-Of-Bag Estimate

Because each tree trains on ~63.2% of the data, the remaining ~36.8% (the OOB set) provides a free validation estimate without needing a separate test split.

- **Definition:** For each training sample, collect predictions only from trees that did NOT train on that sample, then compare to the true label.
- **Formula (classification):**

$$\text{OOB error} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}\left[ \text{majority\_vote}_{t: i \notin B_t}(\hat{y}_t(x_i)) \neq y_i \right]$$

- **Interpretation:** An unbiased estimate of the generalization error. When OOB error stabilizes as more trees are added, training can stop.

**Example from the 3-tree forest above:**

```
OOB predictions collected per sample:
  sample 0 (hours=1, Fail):  Tree 1 → Fail             → vote: Fail  ✓
  sample 1 (hours=2, Fail):  Tree 1 → Fail, Tree 2 → Fail → vote: Fail ✓
  sample 2 (hours=3, Fail):  Tree 1 → Fail             → vote: Fail  ✓
  sample 5 (hours=6, Pass):  Tree 1,2,3 → Pass         → vote: Pass  ✓
  sample 6 (hours=7, Pass):  Tree 3 → Pass             → vote: Pass  ✓
  sample 7 (hours=8, Pass):  Tree 2,3 → Pass           → vote: Pass  ✓
  sample 8 (hours=9, Pass):  Tree 2 → Pass             → vote: Pass  ✓

OOB accuracy = 7/7 = 100%   OOB error = 0%
```

---

### Cost Function: Aggregated Tree Cost

The forest does not optimize a single global cost function. Each tree minimizes its own local criterion on its bootstrap sample. The ensemble cost is evaluated after prediction:

**Classification — using majority vote accuracy:**

$$J_{RF} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}\left[ \hat{y}_{RF}(x_i) \neq y_i \right]$$

**Regression — using MSE of averaged predictions:**

$$J_{RF} = \frac{1}{n} \sum_{i=1}^{n} \left( \hat{y}_{RF}(x_i) - y_i \right)^2$$

Where:

$$\hat{y}_{RF}(x) = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t(x)$$

**Example (regression, query hours = 6.5):**

| Tree | Bootstrap hours (sample)             | Best split | Right leaf mean | Prediction for x=6.5 |
|------|--------------------------------------|------------|-----------------|----------------------|
| 1    | 4,4,5,7,8,8,8,9,10,10               | <= 5.5     | 80.00           | 80.00                |
| 2    | 1,1,3,3,3,4,5,5,7,10               | <= 5.5     | 79.50           | 79.50                |
| 3    | 1,1,2,3,4,5,9,9,9,10               | <= 5.5     | 82.75           | 82.75                |

$$\hat{y}_{RF}(6.5) = \frac{80.00 + 79.50 + 82.75}{3} = 80.75$$

---

### Alternative Notation (Feature Importance)

Feature importance in a Random Forest is defined as the **mean decrease in impurity** (MDI) that each feature contributes across all splits in all trees:

$$\text{Importance}(f) = \frac{1}{T} \sum_{t=1}^{T} \sum_{\text{node } v \text{ splits on } f} \frac{|S_v|}{n} \cdot \Delta\text{Impurity}(v)$$

Where:
- `T` — number of trees
- `|S_v|` — number of samples reaching node v
- `n` — total training samples
- `ΔImpurity(v)` — impurity reduction at node v (Gini Gain or VR)

Importances are normalized to sum to 1 across all features, giving an interpretable percentage.

**Example (2-feature dataset, Gini-based):**

| Feature        | Best Gini Gain per split | Normalized Importance |
|----------------|--------------------------|-----------------------|
| hours (x1)     | 0.4938                   | 61%                   |
| sleep (x2)     | 0.3160                   | 39%                   |
| **Total**      | **0.8098**               | **100%**              |

---

## How Do We Find the Best Forest?

A Random Forest has no parameters learned by gradient descent. The trees are grown independently using the same greedy splitting algorithm as a single Decision Tree, applied to each bootstrap sample with random feature subsampling. The hyperparameters that control the forest are tuned by cross-validation or OOB error monitoring:

- `T` (number of trees) — more trees reduce variance; OOB error curve flattens when adding more trees stops helping.
- `m` (features per split) — `sqrt(p)` for classification, `p/3` for regression are standard starting points.
- `max_depth` — controls individual tree complexity; shallower trees reduce overfitting but may need more trees to compensate.
- `min_samples_leaf` — minimum samples required in a leaf; larger values smooth predictions and reduce overfitting.

---

## Ensemble Aggregation and the Bias-Variance Tradeoff

Random Forests reduce **variance** without significantly increasing **bias**. Each individual tree has low bias (it can fit complex patterns) but high variance (it is sensitive to the specific training sample). Averaging T independent, identically distributed predictors reduces variance by a factor of T:

$$\text{Var}\left(\frac{1}{T}\sum_{t=1}^{T} \hat{y}_t\right) = \frac{\sigma^2}{T} + \frac{T-1}{T} \cdot \rho \cdot \sigma^2$$

Where `σ²` is the variance of a single tree and `ρ` is the pairwise correlation between trees. Random feature selection reduces `ρ`, directly reducing ensemble variance beyond what bootstrap alone achieves.

- A **small forest** (few trees) has high variance — predictions vary with random seed.
- A **large forest** (many trees) has stable predictions — OOB error converges and adding more trees yields no further improvement.

---

## Summary of Key Formulas

| Concept                          | Formula                                                                              |
|----------------------------------|--------------------------------------------------------------------------------------|
| Classification prediction        | y_hat = majority\_vote( tree\_1(x), ..., tree\_T(x) )                               |
| Regression prediction            | y_hat = (1/T) * sum( tree\_t(x) for t in 1..T )                                     |
| OOB error (classification)       | (1/n) * sum( 1[OOB vote != y\_i] )                                                  |
| Ensemble MSE (regression)        | (1/n) * sum( (y\_hat\_RF(x\_i) - y\_i)^2 )                                          |
| Variance reduction by ensemble   | Var(RF) = sigma^2/T + (T-1)/T * rho * sigma^2                                       |
| Feature importance (MDI)         | Importance(f) = (1/T) * sum over trees and nodes splitting on f of weighted delta impurity |
| Features per split (default)     | m = sqrt(p) for classification,  m = p/3 for regression                             |
| Bootstrap sample size            | n samples drawn with replacement; ~63.2% unique, ~36.8% OOB                         |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- Random Forest — Scikit-learn *(coming soon)*

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=J4Wdy0Wc_xQ" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/J4Wdy0Wc_xQ/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=nyxTdL_4Q-Q" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/nyxTdL_4Q-Q/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=sQ870aTKqiM" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/sQ870aTKqiM/hqdefault.jpg"/>
  </a>
</div>