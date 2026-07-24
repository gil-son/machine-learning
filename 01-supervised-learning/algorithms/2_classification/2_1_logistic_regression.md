# Logistic Regression <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

Binary classification | Multi-class classification | Decision boundary

---

## What is Logistic Regression? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

Logistic Regression is a **supervised learning algorithm** used to predict a **categorical outcome** (e.g., pass/fail, spam/not spam, disease/no disease) based on one or more input variables.

Despite its name, it is a **classification** algorithm, not a regression one. At its core, it estimates the **probability** that an input belongs to a given class, then assigns a class label based on a threshold.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to find a function that maps input features `x` to a probability between 0 and 1, by learning a set of weights and a bias term from labeled training data. A **decision boundary** then separates the classes based on that probability.

The key component that makes logistic regression different from linear regression is the **sigmoid function**, which squashes any real-valued number into the range (0, 1):

**Sigmoid Function:**

```
σ(z) = 1 / (1 + e^(-z))
```

Where `z = w * x + b` is the linear combination of inputs — the same expression used in linear regression. The sigmoid wraps it so the output is always a valid probability.

---

## Binary Logistic Regression (1 output class)

Binary Logistic Regression models the probability that an input belongs to class 1 (the positive class), given a single or multiple input features. The output is always between 0 and 1.

**Formula:**

```
y_hat = σ(w * x + b) = 1 / (1 + e^(-(w * x + b)))
```

Also written as:

```
P(y=1 | x) = σ(w^T * x + b)
```

Where:

- `y_hat` — predicted probability of belonging to class 1
- `x` — input feature (independent variable)
- `w` — weight (how strongly the feature influences the prediction)
- `b` — bias or intercept (shifts the decision boundary)
- `σ` — sigmoid function (maps the linear output to a probability)

**Class assignment rule:**

```
if y_hat >= 0.5  →  predict class 1
if y_hat <  0.5  →  predict class 0
```

The threshold 0.5 corresponds to `z = 0`, which is the **decision boundary** — the point where the model is equally uncertain between both classes.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Exam Pass/Fail Based on Study Hours</summary>
  <br/>

  Imagine predicting whether a student will pass or fail an exam based on how many hours they studied.

  **Dataset (Hours Studied vs Pass/Fail):**

  | Hours Studied (x) | Result (y) |
  |-------------------|------------|
  | 1                 | 0 (Fail)   |
  | 2                 | 0 (Fail)   |
  | 3                 | 0 (Fail)   |
  | 4                 | 0 (Fail)   |
  | 5                 | 1 (Pass)   |
  | 6                 | 1 (Pass)   |
  | 7                 | 1 (Pass)   |
  | 8                 | 1 (Pass)   |
  | 9                 | 1 (Pass)   |
  | 10                | 1 (Pass)   |

  **Step 1 — Observe that the output is categorical:**

  Unlike linear regression, the target `y` is not a continuous number — it is a binary label (0 or 1). Plotting hours vs result shows that the data does not follow a straight line; it jumps from 0 to 1 around a threshold. A linear model would predict values outside [0, 1], which cannot be interpreted as probabilities.

  Logistic regression solves this by wrapping the linear output `z = w * x + b` inside the sigmoid function:

  ```
  y_hat = 1 / (1 + e^(-z))
  ```

  This ensures the output is always a valid probability between 0 and 1.

  **Step 2 — The Fitted Model:**

  After training on this data using Gradient Descent, the model learns:

  - `w = 2.9345`
  - `b = -13.0586`

  So the fitted model is:

  ```
  y_hat = σ(2.9345 * x - 13.0586)
  ```

  **Step 3 — Decision Boundary:**

  The decision boundary is the value of `x` where `y_hat = 0.5`, which occurs when `z = 0`:

  ```
  w * x + b = 0
  2.9345 * x - 13.0586 = 0
  x = 13.0586 / 2.9345 ≈ 4.45 hours
  ```

  Students who study more than ~4.45 hours are predicted to pass; those who study less are predicted to fail.

  **Step 4 — Prediction Example:**

  What is the predicted probability of passing for a student who studies 5 hours?

  ```
  z     = 2.9345 * 5 - 13.0586 = 14.6725 - 13.0586 = 1.614
  y_hat = 1 / (1 + e^(-1.614)) = 1 / (1 + 0.199) ≈ 0.834
  ```

  The model predicts an **83.4% probability of passing**. Since 0.834 ≥ 0.5, the predicted class is **1 (Pass)**.

  **Verification against training data:**

  | Hours | z       | Probability | Predicted | Actual |
  |-------|---------|-------------|-----------|--------|
  | 1     | -10.124 | 0.0000      | 0 (Fail)  | 0      |
  | 2     | -7.190  | 0.0008      | 0 (Fail)  | 0      |
  | 3     | -4.255  | 0.0140      | 0 (Fail)  | 0      |
  | 4     | -1.321  | 0.2107      | 0 (Fail)  | 0      |
  | 5     |  1.614  | 0.8340      | 1 (Pass)  | 1      |
  | 6     |  4.549  | 0.9895      | 1 (Pass)  | 1      |
  | 7     |  7.483  | 0.9994      | 1 (Pass)  | 1      |
  | 8     | 10.418  | 1.0000      | 1 (Pass)  | 1      |
  | 9     | 13.352  | 1.0000      | 1 (Pass)  | 1      |
  | 10    | 16.287  | 1.0000      | 1 (Pass)  | 1      |

  All 10 examples are classified correctly. The model is confident at the extremes and uncertain near the decision boundary (~4.45 hours).

  **Visual Analogy:**

  Imagine an S-shaped curve drawn through the scatter plot. At low hours the curve hugs 0; at high hours it hugs 1; in between it rises steeply through 0.5. The point where the curve crosses 0.5 is the **decision boundary** — the dividing line between predicted classes.

</details>

---

## Multi-Feature Logistic Regression (n variables)

Multi-Feature Logistic Regression extends the binary case to handle more than one input feature. Instead of a single weight, the model learns a separate weight for each feature, and the decision boundary becomes a **hyperplane** in the feature space.

**Formula:**

```
y_hat = σ(w1*x1 + w2*x2 + ... + wn*xn + b)
```

Also written as:

```
P(y=1 | x) = σ(w^T * x + b)
```

Each weight `wi` captures the independent contribution of feature `xi` to the log-odds of belonging to class 1, and `b` is the bias. The sigmoid then converts the combined score to a probability.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Exam Pass/Fail Based on Study Hours and Sleep Hours</summary>
  <br/>

  Imagine predicting whether a student will pass an exam based on two features: hours studied and hours of sleep the night before.

  **Dataset:**

  | Hours Studied (x1) | Hours Sleep (x2) | Result (y) |
  |--------------------|------------------|------------|
  | 1                  | 4                | 0 (Fail)   |
  | 2                  | 5                | 0 (Fail)   |
  | 3                  | 4                | 0 (Fail)   |
  | 4                  | 6                | 0 (Fail)   |
  | 5                  | 7                | 1 (Pass)   |
  | 6                  | 6                | 1 (Pass)   |
  | 7                  | 8                | 1 (Pass)   |
  | 8                  | 7                | 1 (Pass)   |
  | 9                  | 8                | 1 (Pass)   |
  | 10                 | 9                | 1 (Pass)   |

  **Step 1 — Identify the features and their roles:**

  Each feature contributes independently to the probability of passing:
  - `x1` (study hours) — more study time generally increases the chance of passing
  - `x2` (sleep hours) — adequate sleep supports retention and focus

  The model needs to learn how much each feature contributes (its weight) and what the baseline log-odds is (the bias).

  **Step 2 — The Fitted Model:**

  After training on this data using Gradient Descent, the model learns:

  - `w1 =  5.9438` — study hours have a strong positive effect on passing probability
  - `w2 = -2.4658` — once study hours are accounted for, additional sleep slightly reduces the score (likely due to collinearity in this small dataset)
  - `b  = -10.7619` — baseline log-odds before features are considered

  The fitted model is:

  ```
  y_hat = σ(5.9438*x1 + (-2.4658)*x2 - 10.7619)
  ```

  > The negative weight for sleep may seem counterintuitive. In a small dataset like this, features can be correlated in ways that produce unexpected signs. With more data and proper regularization, the weight would likely be positive.

  **Step 3 — Prediction Example:**

  What is the predicted probability of passing for a student who studied 5 hours and slept 7 hours?

  ```
  z     = 5.9438*5 + (-2.4658)*7 - 10.7619
        = 29.719 - 17.261 - 10.762
        = 1.696
  y_hat = 1 / (1 + e^(-1.696)) ≈ 0.845
  ```

  The model predicts an **84.5% probability of passing**. Since 0.845 ≥ 0.5, the predicted class is **1 (Pass)**.

  **Verification against training data:**

  | Hours Study | Hours Sleep | Probability | Predicted | Actual |
  |-------------|-------------|-------------|-----------|--------|
  | 1           | 4           | 0.0000      | 0 (Fail)  | 0      |
  | 2           | 5           | 0.0000      | 0 (Fail)  | 0      |
  | 3           | 4           | 0.0577      | 0 (Fail)  | 0      |
  | 4           | 6           | 0.1441      | 0 (Fail)  | 0      |
  | 5           | 7           | 0.8451      | 1 (Pass)  | 1      |
  | 6           | 6           | 1.0000      | 1 (Pass)  | 1      |
  | 7           | 8           | 1.0000      | 1 (Pass)  | 1      |
  | 8           | 7           | 1.0000      | 1 (Pass)  | 1      |
  | 9           | 8           | 1.0000      | 1 (Pass)  | 1      |
  | 10          | 9           | 1.0000      | 1 (Pass)  | 1      |

  All 10 examples are classified correctly. With two features, the decision boundary is no longer a single point on a number line — it becomes a **line** in the 2D feature space that separates the failing region from the passing region.

  **Interpreting the weights:**

  Each weight tells you the isolated effect of that feature on the log-odds of the positive class, assuming all other features are held constant. A positive weight increases the probability of class 1; a negative weight decreases it.

  > **Note:** In practice, weights are learned by minimizing the cost function (Binary Cross-Entropy) over the full training set using Gradient Descent. For large feature spaces, regularization (L1 or L2) is added to prevent overfitting.

</details>

---

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

Logistic Regression assumes a **linear decision boundary** — it can only separate classes that are linearly separable in the feature space. When this assumption does not hold — for example, when classes form concentric circles, or when the relationship between features and outcome is highly non-linear — the model's predictive performance suffers. In such cases, alternatives include Support Vector Machines with kernel functions, Decision Trees, Random Forests, or Neural Networks.

---

## Error and the Cost Function

### Why Not Use MSE for Classification?

Mean Squared Error works well for regression but poorly for classification. When applied to probabilities output by the sigmoid, the MSE cost surface becomes **non-convex** — full of local minima that make Gradient Descent unreliable. Instead, logistic regression uses **Binary Cross-Entropy (Log Loss)**, which is convex and penalizes confident wrong predictions very heavily.

| Predicted Probability | Actual | Loss (wrong direction is expensive) |
|-----------------------|--------|--------------------------------------|
| 0.85                  | 1      | 0.1625 (correct and confident — low loss) |
| 0.20                  | 0      | 0.2231 (correct and fairly confident — low loss) |
| 0.60                  | 1      | 0.5108 (correct but uncertain — moderate loss) |

---

### Objective: Minimize the Error

The model's goal is to find weights and bias such that the predicted probabilities are as close as possible to the true labels (0 or 1) across all training examples.

---

### Binary Cross-Entropy (Log Loss)

- **Definition:** Measures how far the predicted probability is from the true binary label, penalizing confident wrong predictions logarithmically.
- **Formula:**

$$\text{Loss}(y, \hat{y}) = -[y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y})]$$

- **Interpretation:** When the true label is 1, the loss is `-log(y_hat)` — large when y_hat is near 0; when the true label is 0, the loss is `-log(1 - y_hat)` — large when y_hat is near 1.

---

### Cost Function: Binary Cross-Entropy (BCE)

BCE averages the log loss across all training examples, making it comparable across datasets of different sizes.

**Formula:**

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right]$$

Where:
- `m` — number of data points
- `y_hat_i` — predicted probability from the sigmoid
- `y_i` — actual binary label (0 or 1)

**Example Calculation:**

Using the 3-point table above:

- Point 1: y=1, y_hat=0.85 → loss = -log(0.85) = 0.1625
- Point 2: y=0, y_hat=0.20 → loss = -log(1 - 0.20) = 0.2231
- Point 3: y=1, y_hat=0.60 → loss = -log(0.60) = 0.5108

$$J = \frac{0.1625 + 0.2231 + 0.5108}{3} = \frac{0.8964}{3} = 0.2988$$

**Interpretation:** The average log loss across these three predictions is 0.2988. Gradient Descent will adjust `w` and `b` to reduce this value.

---

### Alternative Notation (Cost Function)

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \cdot \log(\sigma(w \cdot x_i + b)) + (1 - y_i) \cdot \log(1 - \sigma(w \cdot x_i + b)) \right]$$

**Example:**

| x_i | y_i (Actual) |
|-----|--------------|
| 1   | 0            |
| 5   | 1            |
| 8   | 1            |

With w = 1.5 and b = -4.0:

**z values and probabilities:**
- i=1: z = 1.5(1) - 4.0 = -2.5  →  y_hat = σ(-2.5) = 0.0759
- i=2: z = 1.5(5) - 4.0 =  3.5  →  y_hat = σ(3.5)  = 0.9706
- i=3: z = 1.5(8) - 4.0 =  8.0  →  y_hat = σ(8.0)  = 0.9997

**Log Loss per point:**
- i=1: y=0  →  -log(1 - 0.0759) = 0.0789
- i=2: y=1  →  -log(0.9707) = 0.0298
- i=3: y=1  →  -log(0.9997) = 0.0003

**BCE:**

$$J(1.5, -4.0) = \frac{0.0789 + 0.0298 + 0.0003}{3} = 0.0363$$

Gradient Descent will adjust w and b to reduce this value.

---

## How Do We Find the Best Weights?

The cost function is minimized using **Gradient Descent** — an iterative optimization method that nudges the weights and bias in the direction that reduces the Binary Cross-Entropy loss.

---

## Gradient Descent

Gradient Descent updates the weights and bias at each iteration by computing how much the cost changes with respect to each parameter (the partial derivative), then stepping in the opposite direction.

**Update Rules:**

$$w := w - \alpha \cdot \frac{\partial J}{\partial w}$$
$$b := b - \alpha \cdot \frac{\partial J}{\partial b}$$

Where `alpha` is the **learning rate** — a small positive number that controls step size.

The partial derivatives for logistic regression are:

$$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_i$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

These have the same form as in linear regression — the only difference is that `y_hat` here comes from the sigmoid, not a plain linear formula.

**Example:**

> Note: The initial values w = 1.5 and b = -4.0 below are chosen for illustration. In practice, weights are typically initialized randomly or to zero.

Using the 10-point study dataset, suppose:
- Partial derivative with respect to w: 0.5804
- Partial derivative with respect to b: 0.1810
- Learning rate alpha = 0.1

**Update weight:**

```
w_new = 1.5 - 0.1 * 0.5804 = 1.5 - 0.0580 = 1.4420
```

**Update bias:**

```
b_new = -4.0 - 0.1 * 0.1810 = -4.0 - 0.0181 = -4.0181
```

After one iteration, the updated parameters are w ≈ 1.442 and b ≈ -4.018. This process repeats — each time recomputing the cost and the gradients — until the cost converges to a minimum or falls below an acceptable threshold.

- A **small learning rate** leads to slow convergence.
- A **large learning rate** may cause the algorithm to overshoot the minimum and diverge.

---

## Summary of Key Formulas

| Concept                  | Formula                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| Sigmoid function         | σ(z) = 1 / (1 + e^(-z))                                                |
| Hypothesis               | y_hat = σ(w * x + b)                                                    |
| Decision boundary        | w * x + b = 0  →  x = -b / w                                           |
| Log Loss (single point)  | Loss = -[y * log(y_hat) + (1-y) * log(1-y_hat)]                        |
| Cost Function (BCE)      | J(w,b) = -(1/m) * sum[y*log(y_hat) + (1-y)*log(1-y_hat)]              |
| Gradient (weight)        | dJ/dw = (1/m) * sum((y_hat - y) * x)                                   |
| Gradient (bias)          | dJ/db = (1/m) * sum(y_hat - y)                                          |
| Gradient update (weight) | w := w - alpha * (dJ/dw)                                                |
| Gradient update (bias)   | b := b - alpha * (dJ/db)                                                |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- Logistic Regression — Scikit-learn *(coming soon)*

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=yIYKR4sgzI8" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/yIYKR4sgzI8/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=xuTiAW0OR40" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/xuTiAW0OR40/hqdefault.jpg"/>
  </a>
</div>