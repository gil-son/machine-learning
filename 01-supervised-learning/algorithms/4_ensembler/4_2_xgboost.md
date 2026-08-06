# XGBoost <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

## What is XGBoost? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

XGBoost (**eXtreme Gradient Boosting**) is a **supervised learning algorithm** used for both **classification** and **regression** tasks. It belongs to the family of **ensemble methods**: instead of relying on a single model, it builds many simple decision trees, one after another, where each new tree focuses on **correcting the mistakes** made by the trees built so far.

At its core, it optimizes a **regularized objective function** using **gradient boosting** — each new tree is fit to the gradients (and, in XGBoost's case, also the second derivatives, or Hessians) of the loss function, rather than to the raw errors.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to build an **additive ensemble** of weak learners (typically shallow decision trees) `f_1, f_2, ..., f_K` that, together, minimize a regularized objective combining the **prediction loss** and a **penalty for tree complexity**, learned incrementally from training data.

---

## Single Boosting Round (1 tree)

A single boosting round models the correction the ensemble should make to its current predictions. It assumes each new tree can nudge the previous prediction closer to the true value by fitting the **residual pattern** left behind.

**Formula:**

```
F_1(x) = F_0(x) + eta * f_1(x)
```

Also written as:

```
y_hat = F_0(x) + eta * f_1(x)
```

Where:

- `y_hat` — updated predicted output
- `F_0(x)` — initial prediction (commonly the mean of `y` for regression)
- `f_1(x)` — the first tree, fit to the residuals of `F_0`
- `eta` — learning rate (shrinkage), controls how much of the new tree's correction is applied

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Student Exam Scores Based on Study Hours</summary>
  <br/>

  Imagine the same dataset used for Linear Regression: predicting exam scores from hours studied.

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

  **Step 1 — Initial Prediction (F0):**

  XGBoost starts with a simple baseline prediction — for regression this is usually the mean of `y`:

  ```
  F0 = (52+55+60+63+66+70+74+78+82+85) / 10 = 68.5
  ```

  **Step 2 — Compute the Residuals:**

  The residual is how far the baseline is from each actual value (`y - F0`):

  | x  | y  | Residual (y - F0) |
  |----|----|--------------------|
  | 1  | 52 | -16.5              |
  | 2  | 55 | -13.5              |
  | 3  | 60 | -8.5               |
  | 4  | 63 | -5.5               |
  | 5  | 66 | -2.5               |
  | 6  | 70 | 1.5                |
  | 7  | 74 | 5.5                |
  | 8  | 78 | 9.5                |
  | 9  | 82 | 13.5               |
  | 10 | 85 | 16.5               |

  **Step 3 — Grow a Tree on the Residuals:**

  A single-split tree is grown to group similar residuals. Suppose the tree splits on `x < 5.5`:

  - **Left node** (x = 1..5): residuals = [-16.5, -13.5, -8.5, -5.5, -2.5], N = 5, sum = -46.5
  - **Right node** (x = 6..10): residuals = [1.5, 5.5, 9.5, 13.5, 16.5], N = 5, sum = 46.5

  **Step 4 — Compute Similarity Scores (with regularization `lambda = 1`):**

  ```
  Similarity = (Sum of Residuals)^2 / (N + lambda)
  ```

  - Root (N=10, sum=0): `Similarity_root = 0^2 / 11 = 0`
  - Left (N=5, sum=-46.5): `Similarity_left = (-46.5)^2 / 6 = 360.375`
  - Right (N=5, sum=46.5): `Similarity_right = (46.5)^2 / 6 = 360.375`

  **Gain of the split:**

  ```
  Gain = Similarity_left + Similarity_right - Similarity_root
       = 360.375 + 360.375 - 0 = 720.75
  ```

  A large positive gain means this split is worth keeping.

  **Step 5 — Compute the Leaf Output Values:**

  ```
  Output Value = Sum of Residuals / (N + lambda)
  ```

  - `Output_left = -46.5 / 6 = -7.75`
  - `Output_right = 46.5 / 6 = 7.75`

  **Step 6 — Update the Predictions (learning rate `eta = 0.3`):**

  ```
  F1(x) = F0 + eta * Output
  ```

  - Students with `x < 5.5`: `F1 = 68.5 + 0.3 * (-7.75) = 66.175`
  - Students with `x >= 5.5`: `F1 = 68.5 + 0.3 * 7.75 = 70.825`

  > **Note:** This is a simplified, single-round view. In practice, XGBoost repeats this process for many rounds (trees), each one correcting the residuals left by the ensemble so far, and the real split search evaluates many candidate thresholds across all features using gradients and Hessians (see the *Objective Function* and *How XGBoost Finds the Best Splits* sections below).

  **Prediction Example:**

  A student who studies 7 hours falls in the `x >= 5.5` group:

  ```
  y_hat = F1 = 70.825
  ```

</details>

---

## Full Additive Ensemble (K trees)

The full model extends the single-round case to many boosting rounds. Instead of one correction, the model learns a sequence of trees, where each one is fit to the residual pattern left by all the trees built before it.

**Formula:**

```
F_K(x) = F_0(x) + eta * f_1(x) + eta * f_2(x) + ... + eta * f_K(x)
```

Also written as:

```
F(x) = F_0(x) + sum_{k=1}^{K} eta * f_k(x)
```

Each tree `f_k` corrects the residuals left after the previous `k-1` trees, scaled down by the learning rate `eta`, and the ensemble sums all contributions plus the initial baseline to produce the final prediction.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting House Prices Across Multiple Boosting Rounds</summary>
  <br/>

  Imagine predicting the sale price of a house using the same dataset as Linear Regression: size (m²), bedrooms, and distance to the city center (km). To keep the tree-building steps readable, this example grows each round's tree using the `Size` feature.

  **Dataset:**

  | Size (x1) | Bedrooms (x2) | Distance km (x3) | Price y (thousands) |
  |-----------|---------------|------------------|---------------------|
  | 50        | 1             | 10               | 150                 |
  | 80        | 2             | 8                | 200                 |
  | 100       | 3             | 5                | 280                 |
  | 120       | 3             | 3                | 340                 |
  | 150       | 4             | 1                | 420                 |

  **Round 0 — Initial Prediction:**

  ```
  F0 = (150+200+280+340+420) / 5 = 278
  ```

  **Round 1:**

  Residuals (`y - F0`): -128, -78, 2, 62, 142

  Split on `Size < 90`:
  - Left (50, 80): N=2, sum=-206 → `Similarity = (-206)^2/(2+1) = 14145.33`
  - Right (100, 120, 150): N=3, sum=206 → `Similarity = (206)^2/(3+1) = 10609.00`
  - Root: N=5, sum=0 → `Similarity = 0`
  - `Gain = 14145.33 + 10609.00 - 0 = 24754.33`

  Output values (`lambda = 1`):
  - `Output_left = -206/3 = -68.67`
  - `Output_right = 206/4 = 51.50`

  Updated predictions (`eta = 0.3`):
  - Size 50, 80: `F1 = 278 + 0.3*(-68.67) = 257.40`
  - Size 100, 120, 150: `F1 = 278 + 0.3*51.50 = 293.45`

  **Round 2:**

  New residuals (`y - F1`): -107.40, -57.40, -13.45, 46.55, 126.55

  Split again on `Size < 90`:
  - Left (50, 80): N=2, sum=-164.80 → `Similarity = (-164.80)^2/3 = 9053.01`
  - Right (100, 120, 150): N=3, sum=159.65 → `Similarity = (159.65)^2/4 = 6372.03`
  - Root: N=5, sum=-5.15 → `Similarity = (-5.15)^2/6 = 4.42`
  - `Gain = 9053.01 + 6372.03 - 4.42 = 15420.62`

  Output values:
  - `Output_left = -164.80/3 = -54.93`
  - `Output_right = 159.65/4 = 39.91`

  Updated predictions (`eta = 0.3`):
  - Size 50, 80: `F2 = 257.40 + 0.3*(-54.93) = 240.92`
  - Size 100, 120, 150: `F2 = 293.45 + 0.3*39.91 = 305.42`

  **Verification after Round 2:**

  | Size | Actual | Predicted (F2) | Error   |
  |------|--------|-----------------|---------|
  | 50   | 150    | 240.92          | -90.92  |
  | 80   | 200    | 240.92          | -40.92  |
  | 100  | 280    | 305.42          | -25.42  |
  | 120  | 340    | 305.42          | +34.58  |
  | 150  | 420    | 305.42          | +114.58 |

  The errors are shrinking round over round but haven't converged yet — with a conservative `eta = 0.3` and only 2 rounds, the model needs several more rounds (and splits on the other features) to close the gap. This is expected: unlike Linear Regression's closed-form solution, XGBoost always converges **iteratively**.

  > **Note:** XGBoost does not have a "Normal Equation" shortcut — trees are always grown iteratively. This is precisely what makes it flexible enough to capture non-linear patterns, at the cost of needing many rounds and careful tuning of `eta`, tree depth, and the number of rounds `K` (often chosen via early stopping on a validation set).

</details>

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

XGBoost can overfit when trees are grown too deep, too many rounds are used, or the learning rate is too high relative to the data — it will keep fitting residuals, including noise. It also has more hyperparameters to tune (`eta`, `max_depth`, `lambda`, `gamma`, `subsample`, `colsample_bytree`, number of rounds) than simpler models, and its predictions are less directly interpretable than a linear model's coefficients. In such cases, alternatives include LightGBM (faster on very large datasets via histogram-based, leaf-wise growth), CatBoost (handles categorical features natively), Random Forest (bagging instead of boosting, generally more robust to noise), or a simpler linear/tree model when interpretability matters more than raw predictive power.

---

## Error and the Objective Function

### Squared Errors

The **error** is the difference between the predicted value (`y_hat`) and the actual value (`y`). Errors are squared to eliminate negatives and to penalize large mistakes more heavily than small ones.

| Prediction | Actual | Error | Squared Error |
|------------|--------|-------|---------------|
| 70         | 75     | -5    | 25            |
| 82         | 80     | +2    | 4             |
| 60         | 50     | +10   | 100           |

**SSE (Sum of Squared Errors)** for this table: `25 + 4 + 100 = 129`

---

### Regularization Term (Ω)

Unlike plain Gradient Boosting, XGBoost adds an explicit **complexity penalty** for each tree, so the model is discouraged from growing overly large or extreme trees.

**Formula:**

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Where:
- `T` — number of leaves in the tree
- `w_j` — output value (weight) of leaf `j`
- `gamma` — minimum gain required to keep a split (controls tree size)
- `lambda` — L2 regularization strength on the leaf weights

**Example:**

Using the single-round tree from earlier, with leaves `w1 = -7.75` and `w2 = 7.75`, `lambda = 1`, `gamma = 0`:

$$\Omega(f) = 0 \cdot 2 + \frac{1}{2}(1)\left[(-7.75)^2 + (7.75)^2\right] = \frac{1}{2}(120.125) = 60.0625$$

---

### Objective Function

$$Obj = \sum_{i=1}^{m} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

**Example:**

Combining the SSE table above with the regularization term computed for that single tree:

$$Obj = 129 + 60.0625 = 189.0625$$

The first term rewards accuracy; the second term penalizes complexity — XGBoost minimizes both together, not accuracy alone.

---

## Objective: Minimize the Regularized Loss

The model's goal is to find, at every boosting round, the tree that most reduces the objective function above — improving predictions while keeping the tree's leaves simple and its weights small.

---

## How XGBoost Finds the Best Splits

Rather than recomputing the full loss for every candidate split, XGBoost uses a **second-order Taylor approximation** of the loss around the current prediction. Each training point contributes a **gradient** `g_i` (first derivative of the loss) and a **Hessian** `h_i` (second derivative), and splits are scored using only the sums of these values in each node — `G = sum(g_i)` and `H = sum(h_i)`.

**Optimal leaf weight:**

$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

**Gain of a candidate split:**

$$Gain = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}\right] - \gamma$$

**Example:**

Reusing the small dataset from the Linear Regression cost-function example (`x = 1,2,3`, `y = 2,4,5`), with predictions `y_hat = 2.0, 3.5, 5.0` from `w=1.5, b=0.5`, and squared-error loss (`g_i = y_hat_i - y_i`, `h_i = 1`):

| i | y_hat | y | g_i  | h_i |
|---|-------|---|------|-----|
| 1 | 2.0   | 2 | 0    | 1   |
| 2 | 3.5   | 4 | -0.5 | 1   |
| 3 | 5.0   | 5 | 0    | 1   |

If all three points sit in one leaf: `G = -0.5`, `H = 3`. With `lambda = 1`:

```
w* = -(-0.5) / (3 + 1) = 0.125
```

**Update to the prediction** (learning rate `eta = 0.1`, mirroring Gradient Descent's `alpha`):

```
delta = eta * w* = 0.1 * 0.125 = 0.0125
```

**Should we split this leaf?** Suppose a candidate split puts point 1 alone (`G_L=0, H_L=1`) and points 2–3 together (`G_R=-0.5, H_R=2`), with `lambda=1, gamma=0.1`:

```
Similarity_L = 0^2 / (1+1) = 0
Similarity_R = (-0.5)^2 / (2+1) = 0.0833
Similarity_root = (-0.5)^2 / (3+1) = 0.0625

Gain = 0.5 * (0 + 0.0833 - 0.0625) - 0.1 = 0.0104 - 0.1 = -0.0896
```

Since the gain is **negative**, this split makes the objective worse once the complexity penalty (`gamma`) is accounted for — XGBoost would **prune** it and keep the leaf unsplit.

- A **small `eta`** (learning rate) leads to slow convergence, needing many more rounds.
- A **large `eta`** may cause the model to overcorrect and overfit quickly.
- A **larger `gamma`** prunes more aggressively, producing smaller, simpler trees.

---

## Summary of Key Formulas

| Concept                 | Formula                                                                                      |
|--------------------------|-----------------------------------------------------------------------------------------------|
| Ensemble Prediction       | F_K(x) = F_0(x) + sum_{k=1}^{K} eta * f_k(x)                                                  |
| Objective Function        | Obj = sum(l(y_i, y_hat_i)) + sum(Omega(f_k))                                                   |
| Regularization            | Omega(f) = gamma*T + (1/2)*lambda * sum(w_j^2)                                                 |
| Optimal Leaf Weight       | w_j* = -G_j / (H_j + lambda)                                                                    |
| Split Gain                | Gain = (1/2) * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)] - gamma               |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- *(coming soon)*

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

Recommended videos:

<div align="center">
  <a href="https://www.youtube.com/watch?v=OtD8wVaFm6E" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/OtD8wVaFm6E/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=8b1JEDvenQU" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/8b1JEDvenQU/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=ZVFeW798-2I" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/ZVFeW798-2I/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=GrJP9FLV3FE" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/GrJP9FLV3FE/hqdefault.jpg"/>
  </a>
</div>