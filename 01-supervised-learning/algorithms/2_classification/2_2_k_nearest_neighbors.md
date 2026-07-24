# K-Nearest Neighbors <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

KNN Classification (1 feature) | KNN Classification (n features) | KNN Regression | Choosing k

---

## What is K-Nearest Neighbors? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

K-Nearest Neighbors (KNN) is a **supervised learning algorithm** used for both **classification** (e.g., pass/fail, spam/not spam) and **regression** (e.g., predicting a score or price) based on the similarity between input examples.

At its core, KNN makes predictions by finding the **k most similar training examples** to a new input and aggregating their labels or values. It is a **lazy learner** — it does not build an explicit model during training; instead, it memorizes the entire training set and does all the computation at prediction time.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to find, for any new input `x`, the `k` training examples closest to it in feature space, then use those neighbors to produce a prediction. The three key decisions are:

- **k** — how many neighbors to consult (controls the bias-variance tradeoff)
- **Distance metric** — how to measure similarity between points (Euclidean, Manhattan, Minkowski)
- **Aggregation rule** — majority vote for classification; mean (or weighted mean) for regression

**Distance Metrics:**

The most common way to measure distance between two points `A = (a1, a2, ..., an)` and `B = (b1, b2, ..., bn)`:

```
Euclidean  d(A,B) = sqrt( (a1-b1)^2 + (a2-b2)^2 + ... + (an-bn)^2 )
Manhattan  d(A,B) = |a1-b1| + |a2-b2| + ... + |an-bn|
Minkowski  d(A,B) = ( |a1-b1|^p + |a2-b2|^p + ... + |an-bn|^p )^(1/p)
```

Where `p` is a parameter: `p=2` recovers Euclidean, `p=1` recovers Manhattan.

**Example (points A=(2,3) and B=(5,7)):**

```
Euclidean  = sqrt((5-2)^2 + (7-3)^2) = sqrt(9+16) = sqrt(25) = 5.0
Manhattan  = |5-2| + |7-3| = 3 + 4 = 7.0
Minkowski (p=3) = (3^3 + 4^3)^(1/3) = (27+64)^(1/3) = 91^(1/3) ≈ 4.498
```

---

## KNN Classification (1 feature)

KNN Classification assigns a class label to a new input by finding the k nearest training points and taking a **majority vote** among their labels.

**Prediction rule:**

```
class = majority_vote( labels of k nearest neighbors )
```

Also written as:

```
y_hat = argmax_c  count(label == c  among k neighbors)
```

Where:

- `y_hat` — predicted class label
- `k` — number of neighbors to consider
- `majority_vote` — the class that appears most often among the k neighbors
- `distance(x, xi)` — a metric measuring how close the query point is to each training point

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Exam Pass/Fail Based on Study Hours</summary>
  <br/>

  Imagine predicting whether a student will pass or fail an exam based on how many hours they studied.

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

  **Step 1 — Compute distances from the query point:**

  A new student studied **4.5 hours**. We measure the absolute distance from 4.5 to every training point:

  | Training x | Label | Distance from 4.5 |
  |------------|-------|-------------------|
  | 4          | Fail  | 0.50              |
  | 5          | Pass  | 0.50              |
  | 3          | Fail  | 1.50              |
  | 6          | Pass  | 1.50              |
  | 2          | Fail  | 2.50              |
  | 7          | Pass  | 2.50              |
  | 1          | Fail  | 3.50              |
  | 8          | Pass  | 3.50              |
  | 9          | Pass  | 4.50              |
  | 10         | Pass  | 5.50              |

  **Step 2 — Select the k nearest neighbors and vote:**

  **k = 1:** nearest neighbor is x=4 (Fail, dist=0.50)
  ```
  Vote: Fail=1, Pass=0  →  predict: Fail
  ```

  **k = 3:** nearest neighbors are x=4 (Fail), x=5 (Pass), x=6 (Pass)
  ```
  Vote: Fail=1, Pass=2  →  predict: Pass
  ```

  **k = 5:** nearest neighbors are x=4 (Fail), x=5 (Pass), x=6 (Pass), x=3 (Fail), x=7 (Pass)
  ```
  Vote: Fail=2, Pass=3  →  predict: Pass
  ```

  **Step 3 — Observe the effect of k:**

  The choice of `k` directly affects the prediction. With k=1 the model relies on a single neighbor and is sensitive to noise. With k=3 or k=5 the vote smooths out individual outliers.

  **Visual Analogy:**

  Imagine plotting all students on a number line by hours studied, colored by pass/fail. For a new point at 4.5, KNN draws a window of the k closest points and asks: "what color dominates inside this window?" That majority color becomes the prediction.

  > **Note:** When k=1, KNN memorizes training data perfectly but often fails on new inputs (overfitting). Very large k smooths the decision boundary but may introduce bias (underfitting). The right k is typically chosen via cross-validation.

</details>

---

## KNN Classification (n features)

KNN with multiple features works exactly the same way — the only change is that **distance is computed across all features simultaneously** using a multi-dimensional metric such as Euclidean distance. The decision boundary becomes a surface in the n-dimensional feature space.

**Prediction rule:**

```
y_hat = majority_vote( labels of k nearest neighbors by Euclidean distance )
```

Also written as:

```
d(x, xi) = sqrt( (x1-xi1)^2 + (x2-xi2)^2 + ... + (xn-xin)^2 )
y_hat = argmax_c  count(label == c  among k neighbors with smallest d)
```

The model finds the k training points with the smallest `d(x, xi)` and votes among their labels.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Exam Pass/Fail Based on Study Hours and Sleep Hours</summary>
  <br/>

  Imagine predicting whether a student will pass based on two features: hours studied and hours of sleep the night before.

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

  **Step 1 — Compute Euclidean distances from the query point:**

  A new student studied **5.5 hours** and slept **6.5 hours**. Query point: `(5.5, 6.5)`.

  Euclidean distance formula:
  ```
  d = sqrt( (x1 - 5.5)^2 + (x2 - 6.5)^2 )
  ```

  | x1 | x2 | Label | Calculation                              | Distance |
  |----|-----|-------|------------------------------------------|----------|
  | 5  | 6   | Fail  | sqrt((5-5.5)^2 + (6-6.5)^2) = sqrt(0.25+0.25) | 0.7071 |
  | 6  | 7   | Pass  | sqrt((6-5.5)^2 + (7-6.5)^2) = sqrt(0.25+0.25) | 0.7071 |
  | 7  | 6   | Pass  | sqrt((7-5.5)^2 + (6-6.5)^2) = sqrt(2.25+0.25) | 1.5811 |
  | 3  | 5   | Fail  | sqrt((3-5.5)^2 + (5-6.5)^2) = sqrt(6.25+2.25) | 2.9155 |
  | 8  | 8   | Pass  | sqrt((8-5.5)^2 + (8-6.5)^2) = sqrt(6.25+2.25) | 2.9155 |
  | 4  | 4   | Fail  | sqrt((4-5.5)^2 + (4-6.5)^2) = sqrt(2.25+6.25) | 2.9155 |
  | 9  | 7   | Pass  | sqrt((9-5.5)^2 + (7-6.5)^2) = sqrt(12.25+0.25) | 3.5355 |
  | 2  | 4   | Fail  | sqrt((2-5.5)^2 + (4-6.5)^2) = sqrt(12.25+6.25) | 4.3012 |
  | 10 | 9   | Pass  | sqrt((10-5.5)^2 + (9-6.5)^2) = sqrt(20.25+6.25) | 5.1478 |

  **Step 2 — Select the k nearest neighbors and vote:**

  **k = 1:** nearest neighbor is (5,6) → Fail
  ```
  Vote: Fail=1, Pass=0  →  predict: Fail
  ```

  **k = 3:** nearest neighbors are (5,6) Fail, (6,7) Pass, (7,6) Pass
  ```
  Vote: Fail=1, Pass=2  →  predict: Pass
  ```

  **k = 5:** nearest neighbors are (5,6) Fail, (6,7) Pass, (7,6) Pass, (3,5) Fail, (8,8) Pass
  ```
  Vote: Fail=2, Pass=3  →  predict: Pass
  ```

  **Interpreting the result:**

  The query point sits right on the boundary between Fail and Pass. With k=1, the single nearest neighbor (which happens to be a Fail) dominates. As k increases, the broader neighborhood — which contains more Pass examples — wins the vote. This illustrates why choosing k matters most near the decision boundary.

  > **Note:** When features have very different scales (e.g., one feature ranges 0–1 and another 0–1000), the larger-scale feature dominates the distance. Always **normalize or standardize features** before applying KNN.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

KNN makes no assumption about the shape of the decision boundary, which makes it flexible — but this comes with costs. It is **slow at prediction time** because distances to all training points must be computed for every new query. It also struggles with **high-dimensional data** (the curse of dimensionality: in many dimensions, all points become roughly equidistant, making "nearest" meaningless). Memory usage grows linearly with the training set. In such cases, alternatives include Logistic Regression, Support Vector Machines, Decision Trees, or approximate nearest-neighbor methods (e.g., KD-trees, Ball trees).

---

## Error and the Distance Function

### Distance Errors

KNN does not minimize a parametric cost function during training. Instead, prediction quality depends on **how well the distance metric captures true similarity** between examples. A poor metric leads to wrong neighbors, which leads to wrong predictions.

| Query | Neighbor | True Distance | Predicted Label | Actual Label | Correct? |
|-------|----------|---------------|-----------------|--------------|----------|
| 4.5   | x=4      | 0.50          | Fail            | —            | —        |
| 4.5   | x=5      | 0.50          | Pass            | —            | —        |
| 4.5   | x=3      | 1.50          | Fail            | —            | —        |

The closer a neighbor, the more influence it should have on the prediction. Weighted KNN addresses this by assigning weight `1/distance` to each neighbor rather than treating all k equally.

---

### Objective: Find the Most Representative Neighbors

The model's goal is to identify the k training examples that are most similar to the new input — where "similar" is defined by the chosen distance metric — and aggregate their outputs into a single prediction.

---

### Choosing k — The Bias-Variance Tradeoff

The value of `k` is the central hyperparameter of KNN. It directly controls how smooth or jagged the decision boundary is.

- **Definition:** `k` is the number of nearest training neighbors used to make each prediction.
- **Effect on classification:**

| k   | Leave-One-Out Accuracy | Behavior                            |
|-----|------------------------|-------------------------------------|
| 1   | 80%                    | Overfits — boundary too jagged      |
| 3   | 100%                   | Well-balanced — smooth boundary     |
| 5   | 100%                   | Well-balanced — smooth boundary     |
| 7   | 50%                    | Underfits — boundary too smooth     |
| 9   | 60%                    | Underfits — nearly all one class    |

- **Interpretation:** Small k makes the model very sensitive to individual training points (low bias, high variance). Large k makes the model insensitive to local structure (high bias, low variance). The right k is found by trying multiple values on a validation set.

---

### KNN Regression — Predicting Continuous Values

KNN can also be used for regression by **averaging the target values** of the k nearest neighbors instead of voting.

**Formula:**

$$\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i$$

Where:
- `y_hat` — predicted continuous value
- `k` — number of nearest neighbors
- `N_k(x)` — the set of k training points closest to query `x`
- `y_i` — the target value of neighbor `i`

**Example:**

Using the study-hours dataset (x → exam score), predict the score for a student who studied **6.5 hours**:

| Training x | Score (y) | Distance from 6.5 |
|------------|-----------|-------------------|
| 7          | 74        | 0.50              |
| 6          | 70        | 0.50              |
| 8          | 78        | 1.50              |
| 5          | 66        | 1.50              |
| 4          | 63        | 2.50              |

**k = 1:** nearest neighbor is x=7, score=74
```
y_hat = 74 / 1 = 74.00
```

**k = 3:** nearest neighbors are x=7 (74), x=6 (70), x=8 (78)
```
y_hat = (74 + 70 + 78) / 3 = 222 / 3 = 74.00
```

**k = 5:** nearest neighbors are x=7 (74), x=6 (70), x=8 (78), x=5 (66), x=4 (63)
```
y_hat = (74 + 70 + 78 + 66 + 63) / 5 = 351 / 5 = 70.20
```

**Interpretation:** With k=3 the prediction is 74.0, reflecting the tight cluster around 6.5 hours. With k=5 it drops to 70.2 as lower-scoring neighbors at x=4 and x=5 are included, pulling the average down.

---

### Alternative Notation (Weighted KNN)

Standard KNN gives equal weight to all k neighbors. **Weighted KNN** assigns more influence to closer neighbors:

$$\hat{y} = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i \cdot y_i}{\sum_{i \in \mathcal{N}_k(x)} w_i}$$

Where the weight of each neighbor is the inverse of its distance:

$$w_i = \frac{1}{d(x, x_i)}$$

**Example (k=3 regression, query x=6.5):**

| Neighbor x | Score y | Distance d | Weight w = 1/d | w * y   |
|------------|---------|------------|----------------|---------|
| 7          | 74      | 0.50       | 2.0000         | 148.00  |
| 6          | 70      | 0.50       | 2.0000         | 140.00  |
| 8          | 78      | 1.50       | 0.6667         | 52.00   |

$$\hat{y} = \frac{148.00 + 140.00 + 52.00}{2.0000 + 2.0000 + 0.6667} = \frac{340.00}{4.6667} \approx 72.86$$

The weighted prediction is **72.86** — the two equidistant neighbors at x=6 and x=7 dominate, and x=8 contributes less because it is farther away.

---

## How Do We Find the Best k?

Unlike parametric models, KNN has no weights to optimize during training — the only parameter to tune is `k`. The standard approach is **k-fold cross-validation**: train on a subset of data, evaluate on a held-out fold, repeat for several values of k, and choose the k with the best average validation performance.

---

## Choosing the Best k

There is no closed-form formula for the optimal k. Instead, it is selected empirically:

**Process:**

1. Try a range of k values (e.g., 1, 3, 5, 7, 9, ...).
2. For each k, evaluate performance using cross-validation (e.g., leave-one-out or k-fold).
3. Plot validation accuracy (classification) or MSE (regression) vs k.
4. Choose the k that minimizes error on the validation set.

**Update Rule (there is none):**

KNN is a non-parametric model — it stores the training data directly and uses it at prediction time. There are no weights, gradients, or iterative updates. The cost of "training" is zero; the cost of prediction is O(n) distance computations per query, where `n` is the number of training examples.

- A **small k** leads to high variance — the model is sensitive to noise and individual outliers.
- A **large k** leads to high bias — the model over-smooths and ignores local structure.

---

## Summary of Key Formulas

| Concept                        | Formula                                                                 |
|--------------------------------|-------------------------------------------------------------------------|
| Euclidean distance             | d = sqrt( sum( (xi - qi)^2 ) )                                          |
| Manhattan distance             | d = sum( \|xi - qi\| )                                                   |
| Minkowski distance             | d = ( sum( \|xi - qi\|^p ) )^(1/p)                                       |
| Classification prediction      | y_hat = majority_vote( labels of k nearest neighbors )                  |
| Regression prediction          | y_hat = (1/k) * sum( y_i  for i in k nearest neighbors )               |
| Weighted regression prediction | y_hat = sum(w_i * y_i) / sum(w_i),  where w_i = 1/d(x, x_i)           |
| Optimal k                      | chosen by cross-validation, minimizing validation error                 |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- K-Nearest Neighbors — Scikit-learn *(coming soon)*

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=HVXime0nQeI" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/HVXime0nQeI/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=4HKqjENq9OU" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/4HKqjENq9OU/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=0p0o5cmgLdE" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/0p0o5cmgLdE/hqdefault.jpg"/>
  </a>
</div>