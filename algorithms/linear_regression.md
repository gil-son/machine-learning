
### <img src="https://cdn-user-icons.flaticon.com/195846/195846442/1744487422513.svg?token=exp=1744488323~hmac=5ae65620d3529f7af31a48da489995dc" width="50"/> Linear Regression

---

## 🔢 What is Linear Regression?

Linear Regression is a **supervised learning algorithm** used to predict a **continuous value** (e.g., price, temperature, score) based on one or more input variables.

At its core, it's about finding the **best-fitting straight line** through the data.

---

## 🧠 The Idea Behind It

You want to find a **line** that best describes the relationship between input features `x` and the output `y`.

---

## 👉 Simple Linear Regression (1 variable)

The formula is:

- ŷ = w * x + b  
- or: f(x) = w * x + b  
- or: y = β₁x + β₀

Where:
- ŷ = predicted output  
- x = input feature (independent variable)  
- w = weight (slope of the line — shows how much y changes per unit change in x)  
- b = bias or intercept (where the line crosses the y-axis — vertical offset)

---

## 📈 Multiple Linear Regression (n variables)

If you have more than one feature:

- ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  
- or: y = β₁x₁ + β₂x₂ + ... + βₙxₙ + β₀

You're summing the weighted contributions of each feature plus a bias.

---

## ❗ Squared Errors

The **error** is the difference between the predicted value (ŷ) and the actual value (y).  
We square it to remove negatives and penalize large errors more heavily.

| Prediction | Actual | Error | Squared Error |
|------------|--------|-------|----------------|
| 70         | 75     | -5    | 25             |
| 82         | 80     | +2    | 4              |
| 60         | 50     | +10   | 100            |

---

## 🎯 Objective: Minimize the Error

We want our predicted values (ŷ) to be as close as possible to the real outputs (y).

---

## 📉 SSE – Sum of Squared Errors

- **Definition**: The total of all squared differences between predicted and actual values.
- **Formula**:  
  $$ SSE = \sum_{i=1}^{m} (ŷᵢ - yᵢ)^2 $$
- **Interpretation**: It gives the total error of your model across all data points, but it's not averaged.

Using the values from the table above (where $m = 3$ data points):

$$ SSE = (70 - 75)^2 + (82 - 80)^2 + (60 - 50)^2 $$
$$ SSE = (-5)^2 + (2)^2 + (10)^2 $$
$$ SSE = 25 + 4 + 100 $$
$$ SSE = 129 $$

**Interpretation of SSE:** The sum of the squared errors for these three predictions is 129.


## 🧮 Cost Function: Mean Squared Error (MSE)

MSE measures the **average squared difference** between predicted and actual values.

### Formula:

$$ MSE = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2 $$

Where:
- $m$ = number of data points
- $\hat{y}_i$ = predicted value
- $y_i$ = actual value

**Example Calculation for MSE:**

Using the SSE calculated above ($SSE = 129$) and the number of data points ($m = 3$):

$$ MSE = \frac{1}{3} \times 129 $$
$$ MSE = 43 $$

**Interpretation of MSE:** The average of the squared errors for these predictions is 43. This gives a sense of the magnitude of the typical error of the model. To understand the error in the original units, we often take the square root of the MSE (Root Mean Squared Error, RMSE), which in this case would be $\sqrt{43} \approx 6.56$. This suggests that, on average, our predictions are off by about 6.56 units.


### 🔍 Example:

Given Squared Errors: 25, 4, and 100

```text
MSE = (25 + 4 + 100) / 3 = 129 / 3 = 43
```

To get the error in the original unit (not squared), take the square root:

```text
√43 ≈ 6.55
```

**Interpretation**: On average, predictions are off by about **6.55 units**.

---

### 💡 Why Square the Errors?

- **To avoid positive and negative errors canceling each other out:**
  - **Example:** Imagine you have two predictions. One is 5 units too high (+5 error), and the other is 5 units too low (-5 error). If you simply added the errors, the total error would be +5 + (-5) = 0. This would incorrectly suggest your model has no error!
  - By squaring the errors, (+5)² = 25 and (-5)² = 25. Now, when you sum them (or average them in MSE), you get a positive value (25 + 25 = 50), accurately reflecting the presence of error in your predictions.

- **To give **more weight** to larger errors:**
  - **Example:** Consider two scenarios:
    - **Scenario 1:** Two errors of 2 units each. Squared errors are 2² = 4 and 2² = 4. Total squared error = 8.
    - **Scenario 2:** One small error of 1 unit and one larger error of 3 units. Squared errors are 1² = 1 and 3² = 9. Total squared error = 10.
  - Notice how the larger error (3 units) contributes disproportionately more to the total squared error (9 out of 10) compared to its simple magnitude relative to the smaller error. This penalization of larger errors is often desirable because significant deviations from the actual values are usually more problematic.

---

### 📐 Alternative Notation (Cost Function):

$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

**Example:**

Let's say we have a simple dataset with one input feature ($x$) and one output ($y$), and we have 3 data points ($m=3$):

| $x_i$ | $y_i$ (Actual) |
|-------|----------------|
| 1     | 2              |
| 2     | 4              |
| 3     | 5              |

Suppose our current linear regression model has a weight $w = 1.5$ and a bias $b = 0.5$. Let's calculate the cost function $J(w, b)$:

1. **Predictions ($\hat{y}_i = w \cdot x_i + b$):**
   - For $i=1$: $\hat{y}_1 = (1.5)(1) + 0.5 = 2$
   - For $i=2$: $\hat{y}_2 = (1.5)(2) + 0.5 = 3.5$
   - For $i=3$: $\hat{y}_3 = (1.5)(3) + 0.5 = 5$

2. **Squared Errors $((\hat{y}_i - y_i)^2)$:**
   - For $i=1$: $(2 - 2)^2 = 0^2 = 0$
   - For $i=2$: $(3.5 - 4)^2 = (-0.5)^2 = 0.25$
   - For $i=3$: $(5 - 5)^2 = 0^2 = 0$

3. **Sum of Squared Errors ($\sum_{i=1}^{m} (\hat{y}_i - y_i)^2$):**
   - $0 + 0.25 + 0 = 0.25$

4. **Mean Squared Error (MSE) or Cost Function $J(w, b)$:**
   - $J(1.5, 0.5) = \frac{1}{3} \times 0.25 = 0.0833$

This value, $0.0833$, represents the current error of our model with the chosen weight and bias. Gradient Descent will aim to adjust $w$ and $b$ to make this cost function value smaller.

---

## 🔄 How Do We Find the Best Weights?

We **minimize the cost function** using **Gradient Descent**.

---

### ⚙️ Gradient Descent

An iterative method to adjust weights and bias to reduce the error.

#### Update Rules:

$$w := w - \alpha \cdot \frac{\partial J}{\partial w}$$
$$b := b - \alpha \cdot \frac{\partial J}{\partial b}$$

**Example:**

**(Important Note: The initial values for weight ($w=1.5$) and bias ($b=0.5$) used in this example are arbitrary and chosen for illustration purposes only. In a real implementation, these values would typically be initialized randomly or to zero.)**

Let's continue with the same example and assume we've calculated the partial derivatives of the cost function with respect to $w$ and $b$ at the current values ($w=1.5, b=0.5$). Let's say we found:

- $\frac{\partial J}{\partial w} = -0.1667$
- $\frac{\partial J}{\partial b} = -0.0833$

And let's assume our learning rate $\alpha$ is $0.1$.

Now, we can apply the update rules:

1. **Update weight ($w$):**
   - $w_{new} = w_{old} - \alpha \cdot \frac{\partial J}{\partial w}$
   - $w_{new} = 1.5 - (0.1) \cdot (-0.1667)$
   - $w_{new} = 1.5 + 0.01667$
   - $w_{new} = 1.51667$

2. **Update bias ($b$):**
   - $b_{new} = b_{old} - \alpha \cdot \frac{\partial J}{\partial b}$
   - $b_{new} = 0.5 - (0.1) \cdot (-0.0833)$
   - $b_{new} = 0.5 + 0.00833$
   - $b_{new} = 0.50833$

After one iteration of Gradient Descent, our new weight is approximately $1.51667$ and our new bias is approximately $0.50833$. We would then recalculate the cost function with these new values. If the cost has decreased, we continue iterating this process, adjusting $w$ and $b$ in the direction that further reduces the error, until we reach a minimum (or a sufficiently low error).

The learning rate $\alpha$ controls how large the steps are in each iteration. A small $\alpha$ might lead to slow convergence, while a large $\alpha$ might cause the algorithm to overshoot the minimum.

---

## ✏️ Summary of Key Formulas

| Concept              | Formula                                                |
|----------------------|---------------------------------------------------------|
| Hypothesis           | $\hat{y} = w \cdot x + b$                               |
| Cost Function (MSE)  | $J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$ |
| Gradient (weight)    | $w := w - \alpha \cdot \frac{\partial J}{\partial w}$     |
| Gradient (bias)      | $b := b - \alpha \cdot \frac{\partial J}{\partial b}$     |

---
