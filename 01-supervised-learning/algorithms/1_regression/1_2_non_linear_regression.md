# Non-Linear Regression <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

Polynomial regression | Exponential regression | Logarithmic regression | When to use non-linear vs linear

---

## What is Non-Linear Regression? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

Non-Linear Regression is a **supervised learning algorithm** used to predict a **continuous value** when the relationship between input and output does not follow a straight line. Instead of fitting a line, the model fits a **curve** that better captures patterns such as acceleration, saturation, or exponential growth.

At its core, it generalizes linear regression by allowing the hypothesis function to be any non-linear form — polynomial, exponential, logarithmic, or others.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is still to find a function that best describes the relationship between input features `x` and the output `y`. The key difference from linear regression is that the function is allowed to curve — the parameters are still learned from data, but the model shape is non-linear.

Non-linear regression models share the same training objective: minimize the error between predicted and actual values. What changes is the hypothesis function used to make predictions.

---

## Polynomial Regression

Polynomial Regression extends linear regression by adding **higher-degree terms** of the input feature. It still fits within the linear regression framework (the model is linear in its parameters), but the relationship with `x` becomes curved.

**Formula:**

```
y_hat = w0 + w1*x + w2*x^2 + w3*x^3 + ... + wn*x^n
```

Also written as:

```
y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
```

Where:

- `y_hat` — predicted output
- `x` — input feature
- `w0, w1, ..., wn` — learned coefficients (weights)
- `n` — the degree of the polynomial (controls how curved the fit is)

The degree `n` is a hyperparameter chosen before training. A degree of 1 reduces to simple linear regression; a degree of 2 produces a parabola; higher degrees produce more complex curves.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Braking Distance Based on Vehicle Speed</summary>
  <br/>

  Imagine predicting how many meters a car needs to brake to a full stop, based on its speed (km/h). Physics tells us that braking distance grows with the square of speed — a classic non-linear relationship.

  **Dataset (Speed vs Braking Distance):**

  | Speed km/h (x) | Braking Distance m (y) |
  |----------------|------------------------|
  | 20             | 6                      |
  | 40             | 18                     |
  | 60             | 40                     |
  | 80             | 72                     |
  | 100            | 110                    |
  | 120            | 160                    |

  **Step 1 — Observe that the relationship is not linear:**

  Check how braking distance changes between each speed step:

  - 6 → 18 (+12)
  - 18 → 40 (+22)
  - 40 → 72 (+32)
  - 72 → 110 (+38)
  - 110 → 160 (+50)

  The increments keep growing — the output accelerates as input increases. A straight line would systematically underfit this curve. A degree-2 polynomial (quadratic) is a natural fit.

  **Step 2 — Build the feature matrix with polynomial terms:**

  For degree 2, each row `x` becomes `[1, x, x^2]`:

  ```
  X = [
    [1,   20,   400],
    [1,   40,  1600],
    [1,   60,  3600],
    [1,   80,  6400],
    [1,  100, 10000],
    [1,  120, 14400]
  ]

  y = [6, 18, 40, 72, 110, 160]
  ```

  **Step 3 — Apply the Normal Equation to find the coefficients:**

  ```
  w = (X^T * X)^-1 * X^T * y
  ```

  The result vector `w` has one entry per column of `X` — in the same order as the columns were built: bias column first, then `x`, then `x^2`.

  **A) Transpose X  →  X^T (3 rows x 6 columns):**

  Each row of `X^T` groups all values of one feature across all data points.

  ```
    [    1       1       1       1       1       1   ]   ← bias column (1s)
    [   20      40      60      80     100     120   ]   ← x (speed)
    [  400    1600    3600    6400   10000   14400   ]   ← x^2 (speed squared)
  ```

  **B) Multiply  X^T * X  →  (3 x 3 matrix):**

  Each cell `[i, j]` is the dot product of row `i` of `X^T` with column `j` of `X^T`
  (which is the same as row `j` of `X^T`), summed across all 6 data points.

  Examples:

  - Cell [0,0] → bias × bias:  1+1+1+1+1+1 = 6
  - Cell [1,1] → x × x:  20²+40²+60²+80²+100²+120² = 400+1600+3600+6400+10000+14400 = 36400
  - Cell [1,2] → x × x²:  20×400 + 40×1600 + 60×3600 + 80×6400 + 100×10000 + 120×14400 = 3528000
  - Cell [2,2] → x² × x²:  400²+1600²+3600²+6400²+10000²+14400² = 364000000

  ```
    [          6        420        36400  ]
    [        420      36400      3528000  ]
    [      36400    3528000    364000000  ]
  ```

  Each cell summarises how two features relate to each other across all 6 data points.
  The matrix is symmetric — cell [i,j] always equals cell [j,i].

  **C) Invert  (X^T * X)^-1  →  (3 x 3 matrix):**

  ```
    [  3.20000000    -0.09750000     0.00062500  ]
    [ -0.09750000     0.00342411    -0.00002344  ]
    [  0.00062500    -0.00002344     0.00000017  ]
  ```

  The inverse isolates the individual contribution of each term — bias, `x`, and `x^2` — by removing the correlations captured in the previous step.

  **D) Multiply  X^T * y  →  (3 x 1 vector):**

  ```
    [      406.00  ]   ← how the bias (1s) correlates with actual distances
    [    39200.00  ]   ← how x (speed) correlates with actual distances
    [  4040000.00  ]   ← how x^2 (speed squared) correlates with actual distances
  ```

  **E) Final result  (X^T * X)^-1 * X^T * y  →  w:**

  ```
    [  2.2000  ]   ← w0  (intercept, from the 1s column)
    [ -0.0475  ]   ← w1  (coefficient for x)
    [  0.0113  ]   ← w2  (coefficient for x^2)
  ```

  - `w0 = 2.2` — the model's baseline offset when speed is 0.
  - `w1 = -0.0475` — the linear correction; slightly reduces the estimate at low speeds.
  - `w2 = 0.0113` — the quadratic term that drives the curve upward as speed grows.

  **The Fitted Model:**

  ```
  y_hat = 0.0113*x^2 - 0.0475*x + 2.2
  ```

  **Step 4 — Prediction Example:**

  What is the predicted braking distance for a car travelling at 90 km/h?

  ```
  y_hat = 0.0113*(90^2) - 0.0475*90 + 2.2
        = 0.0113*8100 - 4.275 + 2.2
        = 91.53 - 4.275 + 2.2
        = 89.45 meters
  ```

  **Verification against training data:**

  | Speed | Actual | Predicted | Error  |
  |-------|--------|-----------|--------|
  | 20    | 6      | 5.79      | -0.21  |
  | 40    | 18     | 18.44     | +0.44  |
  | 60    | 40     | 40.17     | +0.17  |
  | 80    | 72     | 70.97     | -1.03  |
  | 100   | 110    | 110.84    | +0.84  |
  | 120   | 160    | 159.79    | -0.21  |

  The polynomial curve fits the data closely, capturing the acceleration that a straight line would miss.

  **Visual Analogy:**

  Imagine plotting speed vs braking distance on a scatter plot — the points curve upward. A straight line drawn through them would miss most points. A parabola (degree 2) bends with the data, tracking the curve much more accurately.

  > **Note:** Increasing the degree too much leads to **overfitting** — the curve memorizes the training data perfectly but fails on new inputs. Choosing the right degree is critical and is typically validated with a test set.

</details>

---

## Exponential Regression

Exponential Regression models relationships where the output **grows or decays at a rate proportional to its current value**. This is common in biology (population growth), finance (compound interest), and physics (radioactive decay).

**Formula:**

```
y_hat = a * e^(b*x)
```

Also written as:

```
y = a * exp(b * x)
```

Where:

- `y_hat` — predicted output
- `x` — input feature
- `a` — scale factor (value of y when x = 0)
- `b` — growth rate (positive = growth, negative = decay)
- `e` — Euler's number (~2.718)

To fit this model, the equation is often **linearized** by taking the natural log of both sides:

```
ln(y) = ln(a) + b*x
```

This transforms it into a simple linear regression problem in `ln(y)`, allowing standard techniques to be applied.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Bacterial Population Growth Over Time</summary>
  <br/>

  Imagine tracking how a bacterial colony grows over time (hours). Under ideal conditions, bacteria double at a roughly constant rate — a classic exponential pattern.

  **Dataset (Hours vs Colony Size):**

  | Hours (x) | Colony Size (y) |
  |-----------|-----------------|
  | 0         | 100             |
  | 1         | 200             |
  | 2         | 390             |
  | 3         | 800             |
  | 4         | 1600            |
  | 5         | 3200            |

  **Step 1 — Observe that the relationship is exponential:**

  Check the growth ratio between each time step:

  - 100 → 200 (×2.0)
  - 200 → 390 (×1.95)
  - 390 → 800 (×2.05)
  - 800 → 1600 (×2.0)
  - 1600 → 3200 (×2.0)

  The colony roughly doubles each hour — a constant multiplication factor, not a constant addition. This is the signature of exponential growth.

  **Step 2 — Linearize by taking the natural log of y:**

  | Hours (x) | ln(y)  |
  |-----------|--------|
  | 0         | 4.605  |
  | 1         | 5.298  |
  | 2         | 5.966  |
  | 3         | 6.685  |
  | 4         | 7.378  |
  | 5         | 8.071  |

  Now `ln(y)` increases by approximately **+0.693** per hour (which is `ln(2)` — confirming the doubling pattern).

  **Step 3 — Fit a linear regression on ln(y) vs x:**

  Using the linearized data:

  ```
  ln(y_hat) = ln(a) + b*x
  ```

  Solving gives:

  ```
  b     ≈  0.693   (growth rate — equals ln(2), confirming doubling per hour)
  ln(a) ≈  4.605   →  a = e^4.605 ≈ 100
  ```

  **The Fitted Model:**

  ```
  y_hat = 100 * e^(0.693 * x)
  ```

  **Step 4 — Prediction Example:**

  How large will the colony be after 6 hours?

  ```
  y_hat = 100 * e^(0.693 * 6)
        = 100 * e^4.158
        = 100 * 63.9
        ≈ 6390
  ```

  The model predicts approximately **6,390 bacteria** after 6 hours (close to the expected 6,400 from perfect doubling).

  > **Note:** Linearization works well when the data follows a clean exponential pattern. When noise is present, non-linear least squares fitting is preferred, as linearization can distort the error distribution.

</details>

---

## Logarithmic Regression

Logarithmic Regression models relationships where the output **grows quickly at first, then levels off** — the inverse pattern of exponential growth. It is common in learning curves, diminishing returns, and signal intensity.

**Formula:**

```
y_hat = a + b * ln(x)
```

Where:

- `y_hat` — predicted output
- `x` — input feature (must be positive)
- `a` — intercept (baseline value)
- `b` — rate of growth (how fast the output rises before levelling)
- `ln` — natural logarithm

This model is already linear in `ln(x)`, so fitting it requires only transforming the input and applying standard linear regression.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting Skill Level Based on Hours of Practice</summary>
  <br/>

  Imagine tracking how a musician's skill score (0–100) improves with cumulative hours of practice. Early practice produces rapid gains; later practice yields smaller improvements — a diminishing returns pattern.

  **Dataset (Hours of Practice vs Skill Score):**

  | Hours (x) | Skill Score (y) |
  |-----------|-----------------|
  | 1         | 10              |
  | 5         | 30              |
  | 10        | 42              |
  | 50        | 65              |
  | 100       | 77              |
  | 500       | 98              |

  **Step 1 — Observe that the gains are diminishing:**

  - 1 → 5 hours: score goes from 10 to 30 (+20 for 4 extra hours)
  - 5 → 10 hours: score goes from 30 to 42 (+12 for 5 extra hours)
  - 10 → 50 hours: score goes from 42 to 65 (+23 for 40 extra hours)
  - 50 → 100 hours: score goes from 65 to 77 (+12 for 50 extra hours)

  Each doubling of practice hours produces a smaller gain. A straight line would overestimate gains at low hours and underestimate them at high hours.

  **Step 2 — Transform the input by taking ln(x):**

  | ln(x)  | Skill Score (y) |
  |--------|-----------------|
  | 0.000  | 10              |
  | 1.609  | 30              |
  | 2.303  | 42              |
  | 3.912  | 65              |
  | 4.605  | 77              |
  | 6.215  | 98              |

  Now `y` increases in a near-linear fashion with `ln(x)`.

  **Step 3 — Fit linear regression on y vs ln(x):**

  Solving gives approximately:

  ```
  b ≈ 14.0   (gain per unit of ln(x))
  a ≈ 10.0   (baseline score)
  ```

  **The Fitted Model:**

  ```
  y_hat = 10.0 + 14.0 * ln(x)
  ```

  **Step 4 — Prediction Example:**

  What skill score would a musician have after 200 hours of practice?

  ```
  y_hat = 10.0 + 14.0 * ln(200)
        = 10.0 + 14.0 * 5.298
        = 10.0 + 74.17
        = 84.17
  ```

  The model predicts a skill score of approximately **84** after 200 hours.

  **Verification against training data:**

  | Hours | Actual | Predicted | Error |
  |-------|--------|-----------|-------|
  | 1     | 10     | 10.0      |  0.0  |
  | 5     | 30     | 32.5      | +2.5  |
  | 10    | 42     | 42.2      | +0.2  |
  | 50    | 65     | 64.8      | -0.2  |
  | 100   | 77     | 74.5      | -2.5  |
  | 500   | 98     | 97.1      | -0.9  |

  The logarithmic model tracks the diminishing-returns pattern accurately across all orders of magnitude.

</details>

---

## When to Use Non-Linear vs Linear <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

Choosing between linear and non-linear regression is one of the first decisions to make when modelling data. The wrong choice leads to systematic errors that more data or better tuning cannot fix.

### Decision Guide

| Situation | Recommended Model |
|---|---|
| Scatter plot shows a straight trend | Linear Regression |
| Output accelerates or decelerates with input | Polynomial Regression |
| Output roughly doubles/halves per unit of input | Exponential Regression |
| Output grows fast then plateaus (diminishing returns) | Logarithmic Regression |
| Residuals from a linear fit show a curved pattern | Switch to non-linear |
| Data has multiple interacting features | Multiple Linear Regression |
| Relationship is complex with no known form | Tree-based or neural models |

### Practical Signals

**Use linear regression when:**
- The scatter plot shows points arranged along a straight band.
- Residuals (errors) after fitting are randomly scattered — no curve or fan shape.
- The domain implies additive, proportional effects (e.g., salary vs years of experience in a stable field).

**Use polynomial regression when:**
- The scatter plot shows a clear curve with one or two bends.
- The relationship has a known quadratic or cubic form (e.g., projectile motion, braking distance).
- Adding `x^2` or `x^3` terms visibly improves fit without overfitting.

**Use exponential regression when:**
- The output grows or decays by a constant *ratio* per unit of input (not a constant amount).
- A plot of `ln(y)` vs `x` is roughly linear.
- The domain suggests compounding behavior (population, interest, radioactive decay).

**Use logarithmic regression when:**
- Large increases in `x` produce smaller and smaller increases in `y`.
- A plot of `y` vs `ln(x)` is roughly linear.
- The domain suggests diminishing returns (learning curves, sound intensity in decibels, earthquake magnitude).

### A Note on Overfitting

Non-linear models — especially high-degree polynomials — can memorize training data perfectly while failing badly on new data. Always validate on a held-out test set. If train error is very low but test error is high, the model is overfitting and the degree or complexity should be reduced.

---

## Error and the Cost Function

Non-linear regression uses the same error metrics as linear regression. The cost function is still MSE — what changes is only the shape of the hypothesis function.

### Squared Errors

The **error** is the difference between the predicted value (`y_hat`) and the actual value (`y`). Errors are squared to eliminate negatives and penalize large mistakes more heavily.

| Prediction | Actual | Error | Squared Error |
|------------|--------|-------|---------------|
| 73         | 72     | +1    | 1             |
| 42         | 40     | +2    | 4             |
| 155        | 160    | -5    | 25            |

---

### Objective: Minimize the Error

The model's goal is to find parameters (`a`, `b`, `w0...wn`) such that predicted values are as close as possible to actual outputs across all training examples.

---

### Cost Function: Mean Squared Error (MSE)

**Formula:**

$$MSE = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

Where:
- `m` — number of data points
- `y_hat_i` — predicted value from the non-linear model
- `y_i` — actual value

To express error in the original units:

$$RMSE = \sqrt{MSE}$$

---

## How Do We Find the Best Parameters?

For models that can be linearized (logarithmic, exponential after log-transform), the **Normal Equation** provides a closed-form solution after transforming the data.

For polynomial regression, the same Normal Equation applies after constructing the polynomial feature matrix.

For models that cannot be linearized (e.g., `y = a * x^b + c`), **non-linear least squares** methods such as the **Levenberg-Marquardt algorithm** are used — these iteratively minimize the cost function without requiring a linear form.

---

## Gradient Descent for Non-Linear Models

When closed-form solutions are unavailable or too expensive, **Gradient Descent** is used to minimize the cost function iteratively, regardless of the model's shape.

**Update Rule (general form):**

$$\theta_j := \theta_j - \alpha \cdot \frac{\partial J}{\partial \theta_j}$$

Where `theta_j` represents any model parameter (could be `w0`, `w1`, `a`, `b`, etc.) and `alpha` is the **learning rate**.

The partial derivative `dJ/d_theta_j` is computed with respect to the specific non-linear model being fitted. The chain rule is applied when the model contains compositions of functions (e.g., an exponential of a linear combination).

- A **small learning rate** leads to slow but stable convergence.
- A **large learning rate** may cause divergence, especially in non-convex cost surfaces common in non-linear models.

---

## Summary of Key Formulas

| Model                | Formula                                          | Typical Use Case                        |
|----------------------|--------------------------------------------------|-----------------------------------------|
| Polynomial (deg. 2)  | y_hat = w0 + w1*x + w2*x^2                       | Curves with one bend (braking, physics) |
| Polynomial (deg. n)  | y_hat = w0 + w1*x + ... + wn*x^n                 | Complex curves, higher-order patterns   |
| Exponential          | y_hat = a * e^(b*x)                              | Growth/decay at constant ratio          |
| Logarithmic          | y_hat = a + b * ln(x)                            | Diminishing returns, saturation         |
| Cost Function (MSE)  | J = (1/m) * sum((y_hat_i - y_i)^2)              | All models                              |
| Gradient update      | theta := theta - alpha * (dJ/d_theta)            | When closed form is unavailable         |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- Non-linear Regression — Scikit-learn *(coming soon)*

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

<div align="center">
  <a href="https://www.youtube.com/watch?v=nGcMl03LPC0" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/nGcMl03LPC0/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=QptI-vDle8Y" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/QptI-vDle8Y/hqdefault.jpg"/>
  </a>
</div>