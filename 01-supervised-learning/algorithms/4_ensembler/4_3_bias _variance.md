# Bias-Variance Tradeoff <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-bird.png?ref_type=heads" width="5%">

## What is the Bias-Variance Tradeoff? <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-mega-man-thinking-with-coffee.png?ref_type=heads" width="5%">

The Bias-Variance Tradeoff is a core concept for understanding **why models make errors** and how to reduce them. It describes the balance between two sources of error that affect how well a model generalizes to new, unseen data.

At its core, it explains why a model that is too simple **underfits** the data, while a model that is too complex **overfits** it — and how to find the sweet spot between the two.

---

## Components <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-light.png?ref_type=heads" width="8%">

The goal is to build a model whose **total expected error** on unseen data is as low as possible, by balancing two competing sources of error learned from the training data: bias and variance.

---

## Bias (Underfitting)

Bias measures how far, on average, a model's predictions are from the true values. It assumes the relationship the model can capture is too simple to represent the real pattern in the data — that is, error stays high even with more training data, because the model itself lacks the flexibility to fit the underlying trend.

**Formula:**

```
Bias(x) = E[f_hat(x)] - f(x)
```

Also written as:

```
Bias = f_hat_avg(x) - f(x)
```

Where:

- `f_hat(x)` — the model's predicted output
- `E[f_hat(x)]` — the average prediction across many trained versions of the model
- `f(x)` — the true underlying function (the actual relationship)
- `Bias(x)` — how far the average prediction is from the truth

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting House Prices with an Overly Simple Model</summary>
  <br/>

  Imagine the true relationship between house size and price actually curves upward (larger houses gain value faster per square meter), but the model only fits a straight line.

  **Dataset (Size vs Actual Price):**

  | Size (x) | True Price (y) | Line Prediction (y_hat) |
  |----------|-----------------|--------------------------|
  | 50       | 120             | 140                      |
  | 80       | 180             | 175                      |
  | 100      | 230             | 198                      |
  | 120      | 310             | 221                      |
  | 150      | 460             | 256                      |

  **Step 1 — Observe the Pattern of Errors:**

  Compare true price to the line's prediction:

  - 120 vs 140 (+20)
  - 180 vs 175 (-5)
  - 230 vs 198 (-32)
  - 310 vs 221 (-89)
  - 460 vs 256 (-204)

  The errors grow larger and consistently in one direction as size increases — the straight line **systematically underestimates** larger houses. This consistent, directional gap (not random noise) is the signature of **high bias**.

  **Step 2 — Estimate the Bias:**

  Averaging the signed errors gives a sense of the systematic gap:

  ```
  (20 - 5 - 32 - 89 - 204) / 5 = -310 / 5 = -62
  ```

  So: **Bias ≈ -62** (on average, the model underpredicts by about 62 units).

  > **Note:** This is a simplified illustration. In practice, bias is estimated by training the model on many different samples of data and averaging the predictions at each point, then comparing that average to the true value.

  **Visual Analogy:**

  Imagine trying to fit a straight ruler through a curve on a scatter plot:
  - No matter how you angle the ruler, it can never hug the curve.
  - The gap between ruler and curve at every point is the **bias** — it comes from the model's limited shape, not from noisy data.

</details>

---

## Variance (Overfitting)

Variance measures how much a model's predictions change when trained on different samples of data. It assumes the model is highly sensitive to the specific training set it saw, including the noise within it, and therefore fails to generalize well to new data.

**Formula:**

```
Variance(x) = E[(f_hat(x) - E[f_hat(x)])^2]
```

Also written as:

```
Var = E[(f_hat - f_hat_avg)^2]
```

Each squared deviation captures how far a single trained model's prediction strays from the average prediction across many training sets, and `E[...]` averages that spread over all of them.

---

### How it Works <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-rush-curious.png?ref_type=heads" width="5%">

<details>
  <summary>Example: Predicting House Prices with an Overly Flexible Model</summary>
  <br/>

  Imagine training a very flexible model (e.g., a high-degree polynomial) on three slightly different samples drawn from the same population, then predicting the price of a single 100 m² house each time.

  **Predictions for x = 100 across three training samples:**

  | Training Sample | Prediction (y_hat) |
  |------------------|---------------------|
  | Sample A         | 210                 |
  | Sample B         | 340                 |
  | Sample C         | 265                 |

  **Step 1 — Compute the Average Prediction:**

  ```
  E[f_hat(100)] = (210 + 340 + 265) / 3 = 815 / 3 = 271.67
  ```

  **Step 2 — Compute Each Deviation from the Average:**

  - Sample A: 210 - 271.67 = -61.67
  - Sample B: 340 - 271.67 = +68.33
  - Sample C: 265 - 271.67 = -6.67

  **Step 3 — Square and Average the Deviations:**

  ```
  Variance = [(-61.67)^2 + (68.33)^2 + (-6.67)^2] / 3
           = [3803.2 + 4669.0 + 44.5] / 3
           = 8516.7 / 3
           = 2838.9
  ```

  So: **Variance ≈ 2838.9**

  The predictions swing wildly between training samples — from 210 to 340 for the *same* house — which shows the model is chasing noise specific to each sample rather than the underlying trend. This inconsistency is the signature of **high variance**.

  > **Note:** This is a simplified illustration. In practice, variance is estimated the same way — training many models on different samples and measuring how spread out their predictions are at a given point — typically using resampling techniques such as cross-validation or bootstrapping.

  **Visual Analogy:**

  Imagine three flexible wires bent to pass exactly through slightly different sets of scattered points:
  - Each wire looks very different from the others, even though the underlying points come from the same population.
  - That inconsistency between wires — not any single wire's shape — is the **variance**.

</details>

## Limitations and Alternatives <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-dr-wily-smilling.png?ref_type=heads" width="6%">

The Bias-Variance Tradeoff assumes error can be cleanly split into these two sources plus irreducible noise. In practice, reducing one often increases the other — a simpler model lowers variance but raises bias, and a more flexible model does the reverse. In such cases, techniques like regularization (Ridge, Lasso), ensembling (bagging, boosting), or cross-validation help find a better balance than adjusting complexity alone.

---

## Error and the Cost Function

### Bias-Variance Decomposition

The **total expected error** of a model at a point can be decomposed into three parts: the squared bias, the variance, and irreducible noise inherent to the data itself.

| Source              | Description                                             |
|----------------------|---------------------------------------------------------|
| Bias²                | Error from the model being too simple to fit the pattern |
| Variance             | Error from the model being too sensitive to training data |
| Irreducible Error (σ²) | Noise inherent to the data that no model can remove     |

**Why decompose the error this way?**

- **To separate fixable error from unfixable error:** Bias and variance can be reduced by changing the model; irreducible error cannot — it is inherent noise in the data itself.
- **To diagnose model behavior:** A model with high error but low variance across samples points to high bias (underfitting); a model with low bias but wildly different predictions across samples points to high variance (overfitting).

---

### Objective: Balance Bias and Variance

The model's goal is not to eliminate bias or variance individually, but to find the level of model complexity that minimizes their **combined** contribution to total error.

---

### Total Expected Error

- **Definition:** The sum of squared bias, variance, and irreducible error, representing the total expected prediction error on unseen data.
- **Formula:**

$$Error(x) = Bias(x)^2 + Variance(x) + \sigma^2$$

- **Interpretation:** Measures the full expected error at a point, combining systematic and inconsistent sources of error plus unavoidable noise.

Using the earlier examples (Bias ≈ -62, Variance ≈ 2838.9) and assuming irreducible error σ² = 15:

$$Error = (-62)^2 + 2838.9 + 15 = 3844 + 2838.9 + 15 = 6697.9$$

---

### Cost Function: Expected Test Error

Expected Test Error estimates how a model will perform on data it has never seen, by combining the bias-variance decomposition with the noise term.

**Formula:**

$$E[(y - \hat{f}(x))^2] = Bias(\hat{f}(x))^2 + Variance(\hat{f}(x)) + \sigma^2$$

Where:
- `y` — actual value
- `f_hat(x)` — model's prediction
- `Bias(f_hat(x))^2` — squared systematic error
- `Variance(f_hat(x))` — spread of predictions across training sets
- `sigma^2` — irreducible noise

**Example Calculation:**

Using Bias² = 3844 and Variance = 2838.9, with σ² = 15:

$$E[(y - \hat{f}(x))^2] = 3844 + 2838.9 + 15 = 6697.9$$

To express the error in the original units, take the square root:

$$\sqrt{6697.9} \approx 81.84$$

On average, predictions are expected to be off by about **81.84 units** on unseen data.

---

### Alternative Notation (Complexity vs Error)

$$J(complexity) = Bias(complexity)^2 + Variance(complexity)$$

**Example:**

| Model Complexity | Bias² | Variance | Total Error |
|-------------------|-------|----------|--------------|
| Low (linear)       | 3844  | 120      | 3964         |
| Medium (quadratic) | 900   | 900      | 1800         |
| High (degree 9)    | 80    | 2838.9   | 2918.9       |

**Observing the tradeoff:**

- Low complexity: high bias, low variance → underfitting
- Medium complexity: balanced bias and variance → lowest total error
- High complexity: low bias, high variance → overfitting

$$J(medium) = 900 + 900 = 1800$$

Model selection techniques will favor the complexity level that minimizes this total.

---

## How Do We Find the Right Balance?

The total error is minimized by tuning **model complexity** — an iterative process of comparing training and validation performance to find the point where both bias and variance are kept as low as jointly possible.

---

## Model Complexity Control

Model complexity is controlled by adjusting how flexible the model is allowed to be, then evaluating the effect on held-out data at each step.

**Common Techniques:**

$$\text{Regularization: } J(w, b) = MSE + \lambda \sum w_i^2$$
$$\text{Cross-Validation: compare error across k folds to estimate true generalization error}$$

Where `lambda` is the **regularization strength** — a small positive number that penalizes model complexity.

**Example:**

> Note: The values below are chosen for illustration. In practice, the right complexity is found by evaluating multiple candidates on validation data.

Continuing from the example above, suppose three candidate models are compared using 5-fold cross-validation:
- Linear model: average validation error = 3964
- Quadratic model: average validation error = 1800
- Degree-9 polynomial: average validation error = 2918.9

**Selecting the best model:**

```
best_model = min(3964, 1800, 2918.9) = 1800 (Quadratic)
```

The quadratic model is chosen because it achieves the lowest total expected error — it is flexible enough to reduce bias without becoming so sensitive to the training data that variance dominates.

- **Too little complexity** leads to high bias and underfitting.
- **Too much complexity** leads to high variance and overfitting.

---

## Summary of Key Formulas

| Concept                  | Formula                                                              |
|---------------------------|------------------------------------------------------------------------|
| Bias                      | Bias(x) = E[f_hat(x)] - f(x)                                          |
| Variance                  | Variance(x) = E[(f_hat(x) - E[f_hat(x)])^2]                          |
| Total Expected Error      | Error(x) = Bias(x)^2 + Variance(x) + sigma^2                         |
| Regularized Cost Function | J(w,b) = MSE + lambda * sum(w_i^2)                                    |

---

## Code / Notebooks / Projects <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-protoman-with-notebook.png?ref_type=heads" width="9%">

- [Bias-Variance Tradeoff — Scikit-learn](https://github.com/gil-son/machine-learning/blob/main/supervised-learning/notebooks/bias-variance/scikit-learn/Bias_Variance_v1.ipynb)

---

## Recommended Videos <img src="https://gitlab.com/gil-son/useful-images-collection/-/raw/main/png/ml-eddie-dropping-video.png?ref_type=heads" width="5%">

Recommended videos:

<div align="center">
  <a href="https://www.youtube.com/watch?v=EuBBz3bI-aA" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/EuBBz3bI-aA/hqdefault.jpg"/>
  </a>
</div>

---

<div align="center">
  <a href="https://www.youtube.com/watch?v=SjQyLhQIXSM" target="_blank">
      <img width="640" height="360" src="https://i.ytimg.com/vi/SjQyLhQIXSM/hqdefault.jpg"/>
  </a>
</div>