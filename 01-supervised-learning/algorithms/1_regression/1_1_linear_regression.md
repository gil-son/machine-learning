# Linear Regression <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

## What is Linear Regression? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

Linear Regression is a **supervised learning algorithm** used to predict a **continuous value** (e.g., price, temperature, score) based on one or more input variables.

At its core, it finds the **best-fitting straight line** through the data by learning the relationship between input features and a numeric output.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to find a line (or hyperplane, in multiple dimensions) that best describes the relationship between input features `x` and the output `y`, by learning a set of weights and a bias term from training data.

---

## Simple Linear Regression (1 variable)

Simple Linear Regression models the relationship between a single input feature and a continuous output. It assumes the relationship is linear — that is, the output changes at a constant rate as the input changes.

**Formula:**

```
y_hat = w * x + b
```

Also written as:

```
f(x) = w * x + b
y = β₁x + β₀
```

Where:

- `y_hat` — predicted output
- `x` — input feature (independent variable)
- `w` — weight (slope of the line; how much y changes per unit increase in x)
- `b` — bias or intercept (where the line crosses the y-axis when x = 0)

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Student Exam Scores Based on Study Hours</summary>
  <br/>

  Imagine predicting how well a student will score on an exam based on how many hours they study.

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

  **Step 1 — Estimate the Slope:**

  Observe how the score changes for each additional hour:

  - 52 → 55 (+3)
  - 55 → 60 (+5)
  - 60 → 63 (+3)
  - 63 → 66 (+3)
  - 66 → 70 (+4)
  - 70 → 74 (+4)
  - 74 → 78 (+4)
  - 78 → 82 (+4)
  - 82 → 85 (+3)

  On average, the score increases by about **+3.72** per hour studied. This is the **slope (w)**.

  When you increase `x` (hours studied) by 1, the predicted `y` (score) goes up by approximately 3.72. This behavior resembles an arithmetic sequence — the output increases by a near-constant amount for each unit increase in the input, which is the hallmark of a linear trend.

  **Step 2 — Estimate the Intercept:**

  The intercept (b) answers: "What would the predicted score be if no hours were studied?"

  If 1 hour corresponds to a score of 52, and the average increase per hour is ~3.72, we can subtract that to estimate the score at 0 hours:

  ```
  52 - 3.72 = 48.07
  ```

  So: **b ≈ 48.07**

  This means the model predicts that a student who studies 0 hours might still score approximately 48.07 — not because it was observed, but because the trend extrapolates to that value.

  > **Note:** This is a simplified estimation. In practice, the model finds w and b by minimizing prediction error across all data points using the Least Squares Method.

  **The Fitted Model:**

  ```
  y_hat = 3.72 * x + 48.07
  ```

  **Prediction Example:**

  How well will a student do if they study 7.5 hours?

  ```
  y_hat = 3.72 * 7.5 + 48.07 = 27.9 + 48.07 = 75.97
  ```

  **Visual Analogy:**

  Imagine drawing a line through a scatter plot of all data points:
  - The **slope** determines how steep the line is (how fast scores rise with hours).
  - The **intercept** is where the line starts on the y-axis when x = 0.

</details>

---

## Multiple Linear Regression (n variables)

Multiple Linear Regression extends the simple case to handle more than one input feature. Instead of a single weight, the model learns a separate weight for each feature, capturing how each one independently contributes to the predicted output.

**Formula:**

```
y_hat = w1*x1 + w2*x2 + ... + wn*xn + b
```

Also written as:

```
y = β₁x₁ + β₂x₂ + ... + βₙxₙ + β₀
```

Each weight `wi` represents the contribution of feature `xi` to the prediction, and `b` is the bias. The model sums all weighted feature contributions plus the bias to produce the final output.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting House Prices Based on Multiple Features</summary>
  <br/>

  Imagine predicting the sale price of a house using three features: size (in square meters), number of bedrooms, and distance to the city center (in km).

  **Dataset:**

  | Size (x1) | Bedrooms (x2) | Distance km (x3) | Price y (thousands) |
  |-----------|---------------|------------------|---------------------|
  | 50        | 1             | 10               | 150                 |
  | 80        | 2             | 8                | 200                 |
  | 100       | 3             | 5                | 280                 |
  | 120       | 3             | 3                | 340                 |
  | 150       | 4             | 1                | 420                 |

  **Step 1 — Identify the features and their roles:**

  Each feature contributes independently to the price:
  - `x1` (size) — larger houses tend to cost more
  - `x2` (bedrooms) — more bedrooms generally increases value
  - `x3` (distance) — closer to the city center typically means higher price

  The model needs to learn how much each feature contributes (its weight) and what the baseline price is (the bias).

  **Step 2 — How the model learns the weights:**

  The model doesn't guess those values — it computes them mathematically by minimizing the prediction error across all training examples. The closed-form method for this is the **Normal Equation**:

  ```
  w = (X^T * X)^-1 * X^T * y
  ```

  **What is `T`?**

  `T` means **transpose** — it flips a matrix so that rows become columns and columns become rows.

  For example, if `X` has shape 5 rows × 4 columns, then `X^T` has shape 4 rows × 5 columns. This operation is needed so the matrix dimensions align correctly for multiplication.

  To apply it, the dataset is represented as a matrix `X` (with a column of 1s added for the bias) and a vector `y`:

  **What are the `1s`?**

  To learn the bias `b` in the same matrix operation as the weights, a column of 1s is prepended to `X`. This works because the bias is just a weight multiplied by a constant input of 1 — so adding a feature that is always 1 lets the model absorb the bias naturally. Without this column, the formula could only learn weights for real features and the line would be forced through the origin.

  **Building the matrices:**

  ```
  X (5 rows x 4 columns — one row per house, first column is all 1s for bias):

    [  1   50   1   10 ]
    [  1   80   2    8 ]
    [  1  100   3    5 ]
    [  1  120   3    3 ]
    [  1  150   4    1 ]

  y (actual prices):

    [ 150  200  280  340  420 ]
  ```

  **Computing each step:**

  **A) Transpose X  →  X^T (4 rows x 5 columns):**

  ```
    [   1      1      1      1      1   ]
    [  50     80    100    120    150   ]
    [   1      2      3      3      4   ]
    [  10      8      5      3      1   ]
  ```

  **B) Multiply  X^T * X  →  (4 x 4 matrix):**

  ```
    [     5      500      13       27  ]
    [   500    55800    1470     2150  ]
    [    13     1470      39       54  ]
    [    27     2150      54      199  ]
  ```
  
  Each cell is a dot product of a row of X^T with a column of X — it summarises how each feature relates to every other feature across all data points.

  **C) Invert  (X^T * X)^-1  →  (4 x 4 matrix):**

  ```
    [  216.9375    -1.2844    -3.7813   -14.5313 ]
    [   -1.2844     0.0111    -0.1047     0.0828 ]
    [   -3.7813    -0.1047     4.7344     0.3594 ]
    [  -14.5313     0.0828     0.3594     0.9844 ]
  ```

  The inverse "undoes" the scaling and correlation captured in the previous step, isolating the individual contribution of each feature.

  **D) Multiply  X^T * y  →  (4 x 1 vector):**

  ```
    [   1390.0  ]
    [ 155300.0  ]
    [   4090.0  ]
    [   5940.0  ]
  ```

  This captures how each feature (and the bias column) correlates with the actual prices.

  **E) Final result  (X^T * X)^-1 * X^T * y  →  w:**

  ```
    [  298.75  ]   ← b
    [    1.3125]   ← w1 (size)
    [  -15.625 ]   ← w2 (bedrooms)
    [  -20.625 ]   ← w3 (distance)
  ```

  ---

  **The fitted model:**

  - `b  =  298.75` — baseline price when all features are zero
  - `w1 =    1.31` — each additional square meter adds ~1.31k
  - `w2 =  -15.63` — each additional bedroom reduces price by ~15.63k (given the other features)
  - `w3 =  -20.63` — each additional km from center reduces price by ~20.63k

  ```
  y_hat = 1.3125*x1 + (-15.625)*x2 + (-20.625)*x3 + 298.75
  ```

  > The negative weight for bedrooms may seem counterintuitive, but this is normal in multiple regression. Once size is already accounted for, adding a bedroom in the same space can signal smaller rooms, which the model penalises. Each weight reflects the effect of that feature *while holding all other features constant*.

  ---

  **Step 3 — Prediction Example:**

  What is the predicted price for a 110 m² house with 3 bedrooms, 4 km from the center?

  ```
  y_hat = 1.3125*110 + (-15.625)*3 + (-20.625)*4 + 298.75
        = 144.375 - 46.875 - 82.5 + 298.75
        = 313.75 (thousands)
  ```

  The model predicts a price of approximately **$313,750**.

  **Verification against training data:**

  | Size | Bedrooms | Distance | Actual | Predicted | Error |
  |------|----------|----------|--------|-----------|-------|
  | 50   | 1        | 10       | 150    | 142.50    | -7.50 |
  | 80   | 2        | 8        | 200    | 207.50    | +7.50 |
  | 100  | 3        | 5        | 280    | 280.00    |  0.00 |
  | 120  | 3        | 3        | 340    | 347.50    | +7.50 |
  | 150  | 4        | 1        | 420    | 412.50    | -7.50 |

  The model fits well — errors are small and balanced, with no systematic over- or under-prediction.

  **Interpreting the weights:**

  Each weight tells you the isolated effect of that feature, assuming all others are held constant. The sign (positive or negative) tells you the direction of the effect; the magnitude tells you how strong it is.

  > **Note:** The Normal Equation gives an exact solution in one step but becomes slow for very large datasets because matrix inversion is expensive. In those cases, **Gradient Descent** is preferred — it reaches the same weights iteratively rather than all at once.
  
</details>

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

Linear Regression assumes a strictly linear relationship between inputs and outputs. When this assumption does not hold — for example, when data follows a curve, or when irrelevant features or multicollinearity are present — the model's predictive performance suffers. In such cases, alternatives include polynomial regression, regularized models (Ridge, Lasso), or non-linear algorithms.

---

## Error and the Cost Function

### Squared Errors

The **error** is the difference between the predicted value (`y_hat`) and the actual value (`y`). Errors are squared to eliminate negatives and to penalize large mistakes more heavily than small ones.

| Prediction | Actual | Error | Squared Error |
|------------|--------|-------|---------------|
| 70         | 75     | -5    | 25            |
| 82         | 80     | +2    | 4             |
| 60         | 50     | +10   | 100           |

**Why square the errors?**

- **To prevent cancellation:** A +5 error and a -5 error would sum to zero, falsely suggesting no error exists. Squaring both gives 25 + 25 = 50, correctly reflecting the presence of error.
- **To penalize large errors more:** An error of 3 contributes 9 to the total; an error of 1 contributes only 1. This disproportionate weighting encourages the model to avoid large deviations.

---

### Objective: Minimize the Error

The model's goal is to find weights and bias such that the predicted values are as close as possible to the actual outputs across all training examples.

---

### SSE — Sum of Squared Errors

- **Definition:** The total of all squared differences between predicted and actual values.
- **Formula:**

$$SSE = \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

- **Interpretation:** Measures total error across the dataset without averaging.

Using the table above (m = 3):

$$SSE = (-5)^2 + (2)^2 + (10)^2 = 25 + 4 + 100 = 129$$

---

### Cost Function: Mean Squared Error (MSE)

MSE measures the **average squared difference** between predicted and actual values, making it comparable across datasets of different sizes.

**Formula:**

$$MSE = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

Where:
- `m` — number of data points
- `y_hat_i` — predicted value
- `y_i` — actual value

**Example Calculation:**

Using SSE = 129 and m = 3:

$$MSE = \frac{129}{3} = 43$$

To express the error in the original units, take the square root (RMSE):

$$RMSE = \sqrt{43} \approx 6.56$$

On average, predictions are off by about **6.56 units**.

---

### Alternative Notation (Cost Function)

$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

**Example:**

| x_i | y_i (Actual) |
|-----|--------------|
| 1   | 2            |
| 2   | 4            |
| 3   | 5            |

With w = 1.5 and b = 0.5:

**Predictions:**
- i=1: y_hat = 1.5(1) + 0.5 = 2.0
- i=2: y_hat = 1.5(2) + 0.5 = 3.5
- i=3: y_hat = 1.5(3) + 0.5 = 5.0

**Squared Errors:**
- i=1: (2.0 - 2)^2 = 0
- i=2: (3.5 - 4)^2 = 0.25
- i=3: (5.0 - 5)^2 = 0

**MSE:**

$$J(1.5, 0.5) = \frac{0 + 0.25 + 0}{3} = 0.0833$$

Gradient Descent will adjust w and b to reduce this value.

---

## How Do We Find the Best Weights?

The cost function is minimized using **Gradient Descent** — an iterative optimization method that nudges the weights and bias in the direction that reduces error.

---

## Gradient Descent

Gradient Descent updates the weights and bias at each iteration by computing how much the cost changes with respect to each parameter (the partial derivative), then stepping in the opposite direction.

**Update Rules:**

$$w := w - \alpha \cdot \frac{\partial J}{\partial w}$$
$$b := b - \alpha \cdot \frac{\partial J}{\partial b}$$

Where `alpha` is the **learning rate** — a small positive number that controls step size.

**Example:**

> Note: The initial values w = 1.5 and b = 0.5 below are chosen for illustration. In practice, weights are typically initialized randomly or to zero.

Continuing from the example above, suppose:
- Partial derivative with respect to w: -0.1667
- Partial derivative with respect to b: -0.0833
- Learning rate alpha = 0.1

**Update weight:**

```
w_new = 1.5 - 0.1 * (-0.1667) = 1.5 + 0.01667 = 1.51667
```

**Update bias:**

```
b_new = 0.5 - 0.1 * (-0.0833) = 0.5 + 0.00833 = 0.50833
```

After one iteration, the updated parameters are w ≈ 1.517 and b ≈ 0.508. This process repeats — each time recomputing the cost and the gradients — until the cost converges to a minimum or falls below an acceptable threshold.

- A **small learning rate** leads to slow convergence.
- A **large learning rate** may cause the algorithm to overshoot the minimum and diverge.

---

## Summary of Key Formulas

| Concept             | Formula                                                               |
|---------------------|-----------------------------------------------------------------------|
| Hypothesis          | y_hat = w * x + b                                                     |
| Cost Function (MSE) | J(w,b) = (1/m) * sum((y_hat_i - y_i)^2)                              |
| Gradient (weight)   | w := w - alpha * (dJ/dw)                                              |
| Gradient (bias)     | b := b - alpha * (dJ/db)                                              |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- [Simple Linear Regression — Scikit-learn](https://github.com/gil-son/machine-learning/blob/main/supervised-learning/notebooks/simple-linear-regression/scikit-learn/Simple_Linear_Regression_v1.ipynb)

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

Recommended videos:

<div align="center">
  <a href="https://www.youtube.com/watch?v=G9SreIKmRdc" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/G9SreIKmRdc/hqdefault.jpg?sqp=-oaymwFBCOADEI4CSFryq4qpAzMIARUAAIhCGAHYAQHiAQoIGBACGAY4AUAB8AEB-AHUBoAC4AOKAgwIABABGA8gZShkMA8=&rs=AOn4CLCW-cEJh69qWVkhKv7w3xf_j-UU3Q"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=z35xmJ40bgY" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/z35xmJ40bgY/hq720.jpg?sqp=-oaymwEnCNAFEJQDSFryq4qpAxkIARUAAIhCGAHYAQHiAQoIGBACGAY4AUAB&rs=AOn4CLD0FlY2V-_N-gdS2LVoCaEdTzmIsw"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=WCP98USBZ0w" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/WCP98USBZ0w/hqdefault.jpg?sqp=-oaymwEnCOADEI4CSFryq4qpAxkIARUAAIhCGAHYAQHiAQoIGBACGAY4AUAB&rs=AOn4CLAi4w8TxGkYbmMeam_TPHl6GBhyvw"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=Gpd14W4vDIQ" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/Gpd14W4vDIQ/hq720.jpg?sqp=-oaymwEnCNAFEJQDSFryq4qpAxkIARUAAIhCGAHYAQHiAQoIGBACGAY4AUAB&rs=AOn4CLClccV-dhuSPGM3njGpRmhUbf0f7A"/>
  </a>
</div>