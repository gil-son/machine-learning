# Support Vector Machines <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

Linear SVM (1 feature) | Linear SVM (n features) | Kernel SVM | Soft Margin and C

---

## What is a Support Vector Machine? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

A Support Vector Machine (SVM) is a **supervised learning algorithm** used primarily for **classification**, and also for regression (SVR), based on finding the decision boundary that best separates classes with the **largest possible margin**.

At its core, SVM does not just find any boundary that separates classes — it finds the **maximum-margin hyperplane**: the one that is as far as possible from the nearest training point of each class. The training points that sit exactly on the margin edges are called **support vectors**, and they are the only points that determine the final boundary.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to find a hyperplane defined by a weight vector `w` and bias `b` that separates the two classes while maximizing the geometric margin between them. Three concepts are central:

- **Hyperplane** — the decision boundary: `w · x + b = 0`. In 2D this is a line; in 3D a plane; in higher dimensions a hyperplane.
- **Margin** — the total perpendicular distance between the two margin planes (`w · x + b = +1` and `w · x + b = -1`), equal to `2 / ||w||`. SVM maximizes this.
- **Support vectors** — the training points closest to the decision boundary, lying exactly on the margin planes. Only these points influence the position and orientation of the boundary.

**Margin formula:**

```
Margin = 2 / ||w||
```

Where `||w||` is the Euclidean norm (length) of the weight vector:

```
||w|| = sqrt(w1^2 + w2^2 + ... + wn^2)
```

Maximizing the margin is equivalent to minimizing `||w||`, subject to every training point being correctly classified and outside the margin.

---

## Linear SVM (1 feature)

Linear SVM with one feature finds a single threshold value on the number line that separates two classes with the widest possible gap. The decision boundary is a point, and the margin is the distance between the two support vectors.

**Decision function:**

```
f(x) = w * x + b
```

Also written as:

```
f(x) = w · x + b
sign( f(x) ) → class label
```

Where:

- `f(x)` — the signed distance from the decision boundary (positive = class +1, negative = class -1)
- `x` — input feature
- `w` — weight (controls the scale and orientation of the boundary)
- `b` — bias (shifts the boundary along the feature axis)
- `sign(f(x))` — the predicted class: +1 if f(x) ≥ 0, −1 otherwise

**Class assignment rule:**

```
if f(x) >= 0  →  predict class +1  (e.g., Pass)
if f(x) <  0  →  predict class -1  (e.g., Fail)
```

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Classifying Exam Pass/Fail Based on Study Hours</summary>
  <br/>

  Imagine classifying whether a student passes or fails based on how many hours they studied.

  **Dataset (Hours Studied vs Result):**

  | Hours Studied (x) | Label (y) |
  |-------------------|-----------|
  | 1                 | −1 (Fail) |
  | 2                 | −1 (Fail) |
  | 3                 | −1 (Fail) |
  | 4                 | −1 (Fail) |
  | 7                 | +1 (Pass) |
  | 8                 | +1 (Pass) |
  | 9                 | +1 (Pass) |
  | 10                | +1 (Pass) |

  **Step 1 — Identify the support vectors:**

  SVM looks for the two training points closest to the gap between classes. Here:

  - Closest Fail point: **x = 4**
  - Closest Pass point: **x = 7**

  These are the **support vectors**. All other points are farther from the boundary and play no role in defining it.

  **Step 2 — Place the decision boundary at the midpoint:**

  ```
  Decision boundary = (4 + 7) / 2 = 5.5
  ```

  The boundary sits exactly halfway between the two support vectors, maximizing the gap between them.

  **Step 3 — Derive the canonical weight and bias:**

  In canonical SVM form, the margin planes satisfy `f(x_sv) = ±1`. With support vectors at x=4 (y=−1) and x=7 (y=+1):

  ```
  w * 7 + b =  +1   (Pass support vector)
  w * 4 + b =  -1   (Fail support vector)
  ```

  Subtracting:

  ```
  w * (7 - 4) = 2   →   w = 2/3 ≈ 0.6667
  b = 1 - w*7 = 1 - (2/3)*7 = 1 - 14/3 = -11/3 ≈ -3.6667
  ```

  **Step 4 — Compute the margin:**

  ```
  Margin = 2 / ||w|| = 2 / (2/3) = 3.0
  ```

  The margin equals 3 — the geometric gap between x=4 and x=7.

  **The Fitted Model:**

  ```
  f(x) = (2/3)*x - 11/3
  ```

  **Functional margins for all training points:**

  | x  | Label | f(x)    | y * f(x)                    |
  |----|-------|---------|-----------------------------|
  | 1  | −1    | −3.0000 | +3.0000                     |
  | 2  | −1    | −2.3333 | +2.3333                     |
  | 3  | −1    | −1.6667 | +1.6667                     |
  | 4  | −1    | −1.0000 | +1.0000  ← support vector   |
  | 7  | +1    | +1.0000 | +1.0000  ← support vector   |
  | 8  | +1    | +1.6667 | +1.6667                     |
  | 9  | +1    | +2.3333 | +2.3333                     |
  | 10 | +1    | +3.0000 | +3.0000                     |

  Every point satisfies `y * f(x) ≥ 1`, confirming the hard-margin constraint is met.

  **Prediction Example:**

  What class does a student with 5 hours of study get predicted?

  ```
  f(5) = (2/3)*5 - 11/3 = 10/3 - 11/3 = -1/3 ≈ -0.333
  sign(-0.333) = -1  →  predict: Fail
  ```

  And a student with 6 hours?

  ```
  f(6) = (2/3)*6 - 11/3 = 12/3 - 11/3 = 1/3 ≈ +0.333
  sign(+0.333) = +1  →  predict: Pass
  ```

  **Visual Analogy:**

  Imagine plotting students on a number line by hours studied. SVM draws a dividing point (5.5) that creates the widest possible no-man's-land between the two classes. The students at x=4 and x=7 are the sentinels — move either of them and the boundary shifts; move any other student and nothing changes.

  > **Note:** This hard-margin SVM requires the data to be perfectly linearly separable. When classes overlap, the **soft-margin SVM** (with parameter C) allows some violations, trading margin width for fewer misclassifications.

</details>

---

## Linear SVM (n features)

Linear SVM with multiple features finds a **hyperplane** in n-dimensional space that separates two classes with the maximum margin. Each feature gets its own weight, and the decision boundary is the set of all points where the weighted sum equals zero.

**Decision function:**

```
f(x) = w1*x1 + w2*x2 + ... + wn*xn + b
```

Also written as:

```
f(x) = w · x + b
```

Where:

- `f(x)` — signed distance from the decision hyperplane
- `w = [w1, w2, ..., wn]` — weight vector (perpendicular to the hyperplane)
- `b` — bias (shifts the hyperplane)
- `||w||` — norm of the weight vector; margin = `2 / ||w||`

The SVM optimization problem is:

```
minimize   (1/2) * ||w||^2
subject to yi * (w · xi + b) >= 1   for all training points i
```

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Classifying Exam Pass/Fail Based on Study Hours and Practice Score</summary>
  <br/>

  Imagine classifying students as Pass or Fail using two features: hours studied (x1) and practice test score (x2).

  **Dataset:**

  | Hours (x1) | Practice Score (x2) | Label (y)  |
  |------------|---------------------|------------|
  | 2          | 50                  | −1 (Fail)  |
  | 3          | 55                  | −1 (Fail)  |
  | 4          | 58                  | −1 (Fail)  |
  | 5          | 62                  | −1 (Fail)  |
  | 6          | 68                  | +1 (Pass)  |
  | 7          | 73                  | +1 (Pass)  |
  | 8          | 78                  | +1 (Pass)  |
  | 9          | 82                  | +1 (Pass)  |

  **Step 1 — Identify the features and their roles:**

  - `x1` (study hours) — more hours generally correlates with passing
  - `x2` (practice score) — higher practice scores correlate with passing

  SVM will find a line in this 2D space (a hyperplane) that separates the two groups with maximum margin.

  **Step 2 — The Fitted Model:**

  After solving the SVM optimization problem, the model learns:

  - `w1 =  0.0541` — weight for study hours
  - `w2 =  0.3243` — weight for practice score
  - `b  = −21.3784` — bias

  ```
  f(x) = 0.0541*x1 + 0.3243*x2 - 21.3784
  ```

  **Step 3 — Support vectors and margin:**

  The support vectors — the points lying exactly on the margin planes — are:

  - (5, 62) with label −1: `f(5,62) = 0.0541*5 + 0.3243*62 - 21.3784 = −1.000`
  - (6, 68) with label +1: `f(6,68) = 0.0541*6 + 0.3243*68 - 21.3784 = +1.000`

  ```
  ||w|| = sqrt(0.0541^2 + 0.3243^2) = 0.3288
  Margin = 2 / 0.3288 = 6.083
  ```

  **Step 4 — Prediction Example:**

  What is the predicted class for a student with 7 hours of study and a practice score of 73?

  ```
  f(7, 73) = 0.0541*7 + 0.3243*73 - 21.3784
           = 0.3787 + 23.674 - 21.3784
           = +2.674
  sign(+2.674) = +1  →  predict: Pass
  ```

  **Verification against training data:**

  | x1 | x2 | Actual | f(x)    | Predicted |
  |----|-----|--------|---------|-----------|
  | 2  | 50  | −1     | −5.0541 | −1 (Fail) |
  | 3  | 55  | −1     | −3.3784 | −1 (Fail) |
  | 4  | 58  | −1     | −2.3514 | −1 (Fail) |
  | 5  | 62  | −1     | −1.0000 | −1 (Fail) |
  | 6  | 68  | +1     | +1.0000 | +1 (Pass) |
  | 7  | 73  | +1     | +2.6757 | +1 (Pass) |
  | 8  | 78  | +1     | +4.3514 | +1 (Pass) |
  | 9  | 82  | +1     | +5.7027 | +1 (Pass) |

  All points are correctly classified. The two support vectors (5,62) and (6,68) sit exactly on the margin boundaries at ±1.

  **Interpreting the weights:**

  - `w2 >> w1` — practice score contributes far more than study hours to the decision. The hyperplane is nearly vertical when plotted in (x1, x2) space.
  - The larger `||w||` is, the narrower the margin. SVM shrinks `||w||` to widen it.

  > **Note:** When data is not linearly separable in the original feature space, the **kernel trick** maps data into a higher-dimensional space where it becomes separable — without ever computing the coordinates of that space explicitly.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

The hard-margin SVM requires data to be **perfectly linearly separable** in the chosen feature space. When classes overlap or contain noise, it finds no solution at all. The **soft-margin SVM** (controlled by parameter C) relaxes this by allowing some points to violate the margin, trading a wider margin for fewer errors. When even a linear soft-margin boundary is insufficient, **kernel functions** implicitly map data into higher-dimensional spaces where a linear separator exists — at the cost of more hyperparameters to tune (kernel type, gamma, degree). For very large datasets, SVM training can be slow because solving the quadratic optimization problem scales poorly. In those cases, alternatives include Logistic Regression, Gradient Boosted Trees, or Neural Networks.

---

## Error and the Cost Function

### Margin Violations

In the hard-margin SVM every point must satisfy `y * f(x) ≥ 1` — no violations permitted. In the **soft-margin SVM**, slack variables `ξᵢ ≥ 0` measure how much each point violates the margin:

```
yi * f(xi) >= 1 - ξᵢ
```

- `ξᵢ = 0` — point is correctly classified and outside the margin (no violation)
- `0 < ξᵢ ≤ 1` — point is inside the margin but on the correct side
- `ξᵢ > 1` — point is misclassified

This violation is captured by the **Hinge Loss**:

```
Hinge Loss(i) = max(0,  1 − yᵢ * f(xᵢ))
```

| Scenario                 | yᵢ | f(xᵢ) | yᵢ · f(xᵢ) | Hinge Loss           |
|--------------------------|-----|--------|------------|----------------------|
| Correct, confident       | +1  | 2.5    | 2.5        | max(0, 1−2.5) = 0.0  |
| Correct, on margin       | +1  | 1.0    | 1.0        | max(0, 1−1.0) = 0.0  |
| Correct, inside margin   | +1  | 0.4    | 0.4        | max(0, 1−0.4) = 0.6  |
| Wrong prediction         | +1  | −0.8   | −0.8       | max(0, 1+0.8) = 1.8  |

---

### Objective: Maximize Margin While Minimizing Violations

The soft-margin SVM balances two competing goals:

- **Maximize the margin** — keep `||w||` small so the margin `2/||w||` is large
- **Minimize margin violations** — keep the total hinge loss small

---

### Cost Function: Soft-Margin SVM (Hinge Loss + L2 Regularization)

The full soft-margin cost function combines margin maximization with penalized violations:

**Formula:**

$$J(w, b) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{m} \max(0,\ 1 - y_i \cdot f(x_i))$$

Where:
- `(1/2) ||w||^2` — regularization term; minimizing this maximizes the margin
- `C` — regularization parameter controlling the trade-off between margin width and violations
- `max(0, 1 − yᵢ · f(xᵢ))` — hinge loss for point i
- `m` — number of training points

**Effect of C:**

| C value   | Behavior                                                          |
|-----------|-------------------------------------------------------------------|
| Large C   | Penalizes violations heavily → narrow margin, fewer errors       |
| Small C   | Tolerates violations → wide margin, more misclassifications      |

**Example Calculation:**

Using the hinge loss table above (4 points):

$$J = \frac{1}{2}\|w\|^2 + C \cdot \frac{0.0 + 0.0 + 0.6 + 1.8}{4}$$

With `||w|| = 0.6667` (from the 1-feature example) and `C = 1.0`:

$$J = \frac{1}{2}(0.6667)^2 + 1.0 \cdot 0.6 = 0.2222 + 0.6 = 0.8222$$

Optimization will adjust `w` and `b` to reduce this value.

---

### Alternative Notation (Kernel SVM)

When the data is not linearly separable in the original feature space, SVM applies the **kernel trick** — replacing every dot product `xᵢ · xⱼ` with a kernel function `K(xᵢ, xⱼ)` that implicitly computes a dot product in a much higher-dimensional space:

$$K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$$

Where `φ` maps inputs to the higher-dimensional space — but is never computed explicitly.

**Common kernels:**

```
Linear:      K(xA, xB) = xA · xB
Polynomial:  K(xA, xB) = (xA · xB + 1)^d
RBF/Gaussian:K(xA, xB) = exp(-γ * ||xA - xB||^2)
```

**Example (xA = (1,2), xB = (3,4)):**

- Linear:     `K = 1*3 + 2*4 = 11.0`
- Polynomial (d=2): `K = (11 + 1)^2 = 144.0`
- RBF (γ=0.5): `||xA−xB||^2 = (1−3)^2 + (2−4)^2 = 4+4 = 8`
  `K = exp(−0.5 * 8) = exp(−4.0) ≈ 0.0183`

The RBF kernel produces values close to 1 for similar points and close to 0 for distant points — making it a measure of Gaussian similarity. Gradient Descent on the kernel SVM adjusts the dual coefficients (one per support vector) to reduce the cost function.

---

## How Do We Find the Best Weights?

The SVM cost function is minimized using **quadratic programming** (for exact solutions) or **Stochastic Gradient Descent on the hinge loss** (for large-scale problems). Both approaches find the weight vector `w` and bias `b` that maximize the margin while keeping violations within the budget set by `C`.

---

## Gradient Descent for Soft-Margin SVM

For the soft-margin SVM, Gradient Descent (specifically sub-gradient descent, since hinge loss is not differentiable at 0) updates `w` and `b` at each step.

**Update Rules:**

$$w := w - \alpha \cdot \frac{\partial J}{\partial w}$$
$$b := b - \alpha \cdot \frac{\partial J}{\partial b}$$

Where the sub-gradients are:

$$\frac{\partial J}{\partial w} = \begin{cases} w & \text{if } y_i \cdot f(x_i) \geq 1 \text{ (no violation)} \\ w - C \cdot y_i \cdot x_i & \text{if } y_i \cdot f(x_i) < 1 \text{ (violation)} \end{cases}$$

$$\frac{\partial J}{\partial b} = \begin{cases} 0 & \text{if } y_i \cdot f(x_i) \geq 1 \\ -C \cdot y_i & \text{if } y_i \cdot f(x_i) < 1 \end{cases}$$

**Example:**

> Note: The initial values w = 0.3 and b = −1.65 below are chosen for illustration. In practice, weights are typically initialized to zero.

Using the 1-feature dataset, consider the support vector at x=4 (y=−1):

```
f(4) = 0.3 * 4 - 1.65 = -0.45
y * f = (-1) * (-0.45) = 0.45 < 1   →  hinge is active (violation)
```

With C = 1.0 and learning rate α = 0.01:

**Sub-gradient:**

```
dJ/dw = w - C*y*x = 0.3 - 1.0*(-1)*4 = 0.3 + 4 = 4.3
dJ/db = -C*y       = -1.0*(-1)         = 1.0
```

**Update weight:**

```
w_new = 0.3 - 0.01 * 4.3 = 0.3 - 0.043 = 0.257
```

**Update bias:**

```
b_new = -1.65 - 0.01 * 1.0 = -1.65 - 0.01 = -1.660
```

After one step the weight vector moves to push the boundary away from the support vector, increasing the margin. This process repeats for each training point until convergence.

- A **small learning rate** leads to slow but stable convergence.
- A **large learning rate** may cause the algorithm to oscillate and never converge.
- A **large C** makes the gradient steps larger on violated points, driving the model to classify them correctly at the cost of a narrower margin.

---

## Summary of Key Formulas

| Concept                       | Formula                                                                          |
|-------------------------------|----------------------------------------------------------------------------------|
| Decision function             | f(x) = w · x + b                                                                 |
| Class prediction              | sign(f(x)): +1 if f(x) ≥ 0, −1 otherwise                                        |
| Hard-margin constraint        | yᵢ · (w · xᵢ + b) ≥ 1  for all i                                                |
| Margin                        | 2 / \|\|w\|\|                                                                    |
| Optimization objective        | minimize (1/2) \|\|w\|\|^2  subject to hard-margin constraints                   |
| Hinge loss (single point)     | max(0, 1 − yᵢ · f(xᵢ))                                                          |
| Soft-margin cost function     | J = (1/2)\|\|w\|\|^2 + C · sum( max(0, 1 − yᵢ · f(xᵢ)) )                       |
| Gradient update (weight)      | w := w − α · (dJ/dw)                                                             |
| Gradient update (bias)        | b := b − α · (dJ/db)                                                             |
| Linear kernel                 | K(xA, xB) = xA · xB                                                              |
| Polynomial kernel             | K(xA, xB) = (xA · xB + 1)^d                                                     |
| RBF kernel                    | K(xA, xB) = exp(−γ · \|\|xA − xB\|\|^2)                                         |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- Support Vector Machines — Scikit-learn *(coming soon)*

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=efR1C6CvhmE" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/efR1C6CvhmE/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=_YPScrckx28" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/_YPScrckx28/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=05VABNfa1ds" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/05VABNfa1ds/hqdefault.jpg"/>
  </a>
</div>