# Day 4 Notes — Linear (and Logistic) Regression
### CPSC 483 · Wed 6/2 · Dr. Panangadan · 70 slides · (Géron Ch. 4)

> The whole arc: define a regression model → fit it (least squares / normal equation) → judge it (R²) → make it flexible (polynomial features) → fit it a *second way* (gradient descent) → diagnose what goes wrong (under/overfit, bias-variance) → fix overfitting (Ridge / LASSO / Elastic Net) → bend the same machinery into a *classifier* (logistic regression).
>
> Mental model for the day: **one model family, two ways to fit it, one way to keep it honest.**

---

## 1. What a regression model is (slides 3–7)

A **model** is just a representation of a phenomenon that describes the relationship between variables. A **mathematical model** does it numerically. The course's recurring mantra: *"All models are wrong, but some are useful."*

A **regression model** relates one **continuous** dependent variable `y` to one or more independent/explanatory variables `x`, captured by a single equation. The defining feature vs. classification: the target is continuous.

Two-step recipe for *any* modeling (slide 5), worth memorizing because it frames the entire course:
1. **Select the model family** — the generic shape of the relationship (linear, quadratic, …).
2. **Fit the model** — assign values to the parameters so you pick the single "best" member of that family.

**Taxonomy (slide 6):** Simple (one explanatory variable) vs. Multiple (many); each can be Linear or Non-Linear.

**Where it sits in supervised learning (slide 7):** given training pairs, you learn `f: X → Y`. Continuous output → regression ("find the function that fits the data best"). Discrete output → classification ("find the decision boundary that separates groups"). Same learning skeleton, different output type — keep this distinction crisp; it's a classic exam framing question.

---

## 2. The linear regression model (slides 9–11)

The prediction:

```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + … + θₙxₙ
```

- `ŷ` = predicted value
- `n` = number of features
- `xᵢ` = the i-th feature value
- `θⱼ` = the j-th parameter. `θ₀` is the **bias/intercept**; `θ₁…θₙ` are the **feature weights**.

**Vector form (slide 11):**

```
ŷ = h_θ(x) = θ · x = θ₀x₀ + θ₁x₁ + … + θₙxₙ
```

The trick that makes this clean: prepend `x₀ = 1` to every instance. Then the bias term `θ₀` just rides along as another weight, and the whole thing is a single dot product. This is *why* you see that column of 1s in the design matrix.

**Matrix form for the whole dataset:** stack every instance as a row in `X` (with the leading column of 1s) and you get `ŷ = Xθ`. The housing example (slide 10) makes it concrete: each row is one house, columns are size/bedrooms/bathrooms/year, and each row's equation is the same `θ` applied to that house's features.

---

## 3. Fitting by least squares (slides 12–16)

"Which line is better?" is answered by a **cost function**. For linear regression that's the **Mean Squared Error**:

```
MSE(θ) = (1/m) Σᵢ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)²    where ŷ⁽ⁱ⁾ = θᵀx⁽ⁱ⁾
```

- The differences `(ŷ − y)` are **errors / residuals**.
- We **square** them (so positive and negative errors don't cancel, and big errors are punished more).
- **Least squares** = find the θ that minimizes the sum of squared residuals. Graphically (slide 14): minimize `Σ εᵢ² = ε₁² + ε₂² + …`, the vertical gaps between each point and the line.

### Normal equation (slide 15) — the *analytical* solution

There's a closed-form answer — plug in, get θ directly, no iteration:

```
θ̂ = (XᵀX)⁻¹ Xᵀ y
```

`θ̂` is the weight vector that minimizes MSE. Predict with `θ̂ᵀx_new`.

⚠️ **Interpolation vs. extrapolation:** predictions are trustworthy only when `x_new` is *within the range* of the training data (interpolation). Going outside that range (extrapolation) is unreliable — this is exactly why the slide-22/40 classwork asks you to predict `displacement = 600` (likely outside the training range) as a cautionary case.

### In scikit-learn (slide 16)

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_      # θ₀ and θ₁…θₙ
lin_reg.predict(X_new)
```

(Géron note: sklearn's `LinearRegression` actually uses **SVD / pseudoinverse**, not the literal `(XᵀX)⁻¹`, because the pseudoinverse is more numerically stable and works even when `XᵀX` isn't invertible.)

---

## 4. Is the fit any good? — R² (slides 17–21)

The **coefficient of determination R²** = the proportion of total variation in `y` explained by the regression. Built from three sums of squares:

| Term | Formula | Meaning |
|---|---|---|
| **SSR** (regression) | Σ(ŷ − ȳ)² | **explained** variation |
| **SSE** (error) | Σ(ŷ − y)² | **unexplained** variation |
| **SST** (total) | Σ(y − ȳ)² | total variation, and `SST = SSR + SSE` |

```
R² = SSR/SST = 1 − SSE/SST
```

- Ranges 0→1; higher = more variation explained = (usually) better fit.
- ⚠️ **Watch the SSE definition carefully on the exam:** SSE uses `(ŷ − y)` — prediction vs. *actual*. SSR uses `(ŷ − ȳ)` — prediction vs. the *mean*. Mixing these up is the #1 R² mistake.

```python
lin_reg.score(X, y)                    # returns R²
# or
from sklearn.metrics import r2_score
r2_score(y, y_predict)
```

---

## 5. Making linear regression nonlinear (slides 24–28)

You can model curves *with a linear model* by engineering nonlinear features. The model stays linear **in the parameters** — that's all "linear regression" requires.

- Add `z = x²`, then fit `y = θ₀ + θ₁x + θ₂z`, which is `θ₀ + θ₁x + θ₂x²`. Still linear in θ.
- Cross terms work too: `z = x₁·x₂`.

```python
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X)
```

`PolynomialFeatures(degree=3)` on features `a, b` generates: `a², a³, b², b³, ab, a²b, ab²`.

⚠️ **Combinatorial explosion:** the feature count blows up fast as degree and feature count grow. This is the setup for overfitting later in the deck.

---

## 6. Adjusted R² (slides 29–30) — a key "gotcha" concept

> **Will adding more independent variables always increase R²? Even random ones? — Surprisingly, YES.**

Plain R² never *decreases* when you add features, even garbage ones. So a high R² can be an illusion of fit (the slide's joke: you could "predict" the S&P 500 with enough random variables). **Adjusted R²** penalizes for the number of variables `p`:

- `p` = number of explanatory variables (excluding intercept)
- `n` = sample size

It reduces R² for more complex models, so it only rises if a new feature earns its keep. This is the deck's first warning shot about **overfitting**, framed as "a particular problem in Big Data."

---

## 7. Computational complexity of the analytical approach (slide 31)

- The expensive step is inverting `XᵀX`, an `(n+1)×(n+1)` matrix → complexity ~**O(n²)** in the number of **features** (sklearn's pseudoinverse approximation).
- → Gets very slow as features grow large (think 100,000).
- ✅ But it's **linear in the number of instances (m)** — it handles lots of rows fine, as long as they fit in memory.
- ✅ **Predictions are fast** — linear in both instances and features.

This complexity profile is *exactly* the motivation for the next section: when features are huge or data won't fit in memory, switch to gradient descent.

---

## 8. Gradient descent — the *iterative* solution (slides 32–40)

### The idea (slides 33–35)

**Gradient descent** (a.k.a. steepest descent) is a generic iterative optimizer: find a function's min/max using the **first-order derivative**. Géron's image: you're lost in fog on a mountain; feel the steepest downhill slope under your feet and step that way.

- It reaches the **global** optimum if the function has only **one** optimum; otherwise it can settle in a **local** optimum.
- ✅ **Linear regression's MSE is convex** — a single bowl, no local minima. So GD is guaranteed to approach the global minimum (given a reasonable learning rate and enough time). This is *the* reason GD is safe for linear regression.

**Algorithm:**
1. Start with initial weights (e.g., `θ₀ = 0, θ₁ = 0`, or random).
2. Repeatedly nudge θ to reduce MSE.
3. Stop when a convergence condition is met.

**Update step** (the formula to know cold):

```
θ ← θ − η ∇MSE(θ)
```

where the gradient is `∇MSE(θ) = (2/m) Xᵀ(Xθ − y)`.

- `η` (eta) = **learning rate** hyperparameter, controls step size.
- ⚠️ Too small → painfully slow convergence. Too large → overshoots, can **diverge** (bounces to higher cost). The middle-porridge value is what grid search hunts for.
- Each full pass over the training set = one **epoch**.

### The three flavors (slides 36–37) — common exam compare/contrast

| Variant | Data used per step | Trade-off |
|---|---|---|
| **Batch GD** | ALL training samples every step | Stable direction, accurate; slow & memory-heavy on big data |
| **Stochastic GD (SGD)** | ONE random sample per step | Very fast, memory-efficient, good for big/online/adaptive learning; noisy path |
| **Mini-batch GD** | A small fixed/random batch | Compromise; can find global min but may briefly head wrong direction |

SGD's noise is a feature, not just a bug — it can help escape shallow traps, and you typically shuffle and re-batch each epoch.

### Least Squares vs. Gradient Descent (slide 38) — likely a direct exam question

| Least Squares (analytical) | Gradient Descent (iterative) |
|---|---|
| Simple formula, simple to implement | Fast even on big data (just dot products) |
| Big data → huge matrix → lots of memory | SGD is very memory-efficient |
| Matrix inverse expensive for many features | Easily extends to other models |
| Exact (one shot) | Needs tuning: convergence criterion, learning rate |

One-liner to remember: **few features + fits in memory → normal equation; many features or huge/streaming data → gradient descent.**

### Feature scaling (slide 39) ⚠️

When features have very different scales, the MSE bowl becomes a long elongated valley, and GD zig-zags slowly toward the minimum. **Scale your features (e.g., `StandardScaler`) before GD** so the bowl is round and convergence is fast. (Note: the normal equation does *not* need scaling — this requirement is specific to gradient descent.)

```python
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01)
```

(`eta0` = initial learning rate; `penalty=None` = no regularization yet.)

---

## 9. Underfitting, overfitting, bias-variance (slides 41–48)

**Two conflicting objectives** (the "Principle of Parsimony"):
- **Goodness-of-fit** — match the data.
- **Simplicity** — keep the model simple.
- Goal: as simple as possible *without* sacrificing too much fit.

**Overfitting** — model too complex; fits training data (even its *noise*) too closely; poor predictions on new data.
- → **High variance**: low training error, high generalization error. The model memorized noise.
- **Occam's Razor**: equal generalization error → prefer the simpler model.

**Underfitting** — model too simple to capture the structure.
- → **High bias**: high training error *and* high systematic error. Misses real relationships.

### Bias-Variance decomposition (slides 47–48) — memorize the three pieces

Generalization error = **Bias² + Variance + Irreducible error**

- **Bias** — error from wrong assumptions (e.g., assuming linear when it's quadratic). High bias → underfit.
- **Variance** — sensitivity to training-data fluctuations (e.g., high-degree polynomial). High variance → overfit.
- **Irreducible error** — noise in the data; nothing the model can do.

**The trade-off:** ↑complexity → ↑variance, ↓bias. ↓complexity → ↑bias, ↓variance. The job is to find the complexity that generalizes best. (Learning curves are the diagnostic tool — slide 46: a large train/validation gap = overfitting/high variance; both curves high and close = underfitting/high bias.)

---

## 10. Regularization — taming overfitting (slides 49–62)

Idea: add a **penalty term** to the loss that punishes large coefficients, forcing a simpler model:

```
l'(w) = l(w) + α·R(w)
```

- `l(w)` = the original loss (MSE).
- `α > 0` = the tuning/shrinkage strength. **`α = 0` → plain linear regression. Larger α → coefficients pushed toward 0 → simpler model.** Choose α by **cross-validation** — it controls the fit-vs-simplicity trade-off.
- *Why penalize coefficient magnitude?* Large weights = wiggly, high-variance fits; shrinking them smooths the model.

### The three penalties

| Method | Penalty term | Norm | Signature behavior |
|---|---|---|---|
| **Ridge** | `α Σ wᵢ²` | L2 | Shrinks coefficients *toward* zero, never exactly to zero |
| **LASSO** | `α Σ \|wᵢ\|` | L1 | Can shrink coefficients *all the way to* zero → feature selection / sparsity |
| **Elastic Net** | combo of L1 + L2 | both | Blend of the two |

### Ridge (L2) (slides 54–55)
- Has an **analytical solution**: `ŵ = (XᵀX + NαI)⁻¹ Xᵀy`. (It's the normal equation with `+NαI` added — this also makes the matrix always invertible, a nice side benefit.)
- GD update gains a shrink factor: `wⱼ ← (1 − 2α)wⱼ − α·(gradient term)`. Notice the `(1 − 2α)` literally shrinks the weight every step.
- As α↑, coefficients shrink asymptotically toward 0 but never vanish.

### LASSO (L1) (slides 56–57)
- Penalty `α Σ |wᵢ|`. **No analytical solution** (`|w|` isn't differentiable at 0) → solved by **coordinate descent** (optimize one parameter at a time, others fixed, using subgradients).
- Drives some coefficients exactly to **zero** → automatic **feature selection** / sparsity.

### Ridge vs. LASSO (slides 58–60) — frequent exam contrast
- The geometric story: LASSO's constraint region is a **diamond** (corners on the axes); when the loss contour hits a corner, that coefficient becomes exactly 0. Ridge's region is a **circle** (no corners) → shrinks but doesn't zero out.
- For the same α, LASSO gives smaller coefficients and higher RSS than Ridge; many coefficients hit zero (sparsity).
- **Correlated predictors:** Ridge keeps them *similar*; LASSO tends to pick *one* and zero the rest.
- LASSO is great under high multicollinearity and for feature selection, but can be **unstable** (too data-dependent).
- Tuning tip: start with a relatively large α, then decrease slowly.

### Elastic Net (slide 61)
- Combines L1 + L2: `Pα(ŵ) = ((1−α)/2)·‖ŵ‖₂² + α·‖ŵ‖₁`.
- `α = 1` → pure LASSO; `α → 0` → approaches Ridge; in between, interpolates. Gets both feature selection (L1) and the stability of L2.

```python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
```

> ⚠️ Naming clash to watch: the *Elastic Net* mixing parameter (L1-vs-L2 balance, slide 61's "α") is a **different** knob from the overall regularization strength α elsewhere in the deck. In sklearn, `ElasticNet` calls the strength `alpha` and the mix `l1_ratio`. Don't conflate them.

---

## 11. Logistic Regression — regression machinery → classification (slides 63–68)

This is the bridge back to Day 3's classification material.

**Idea:** take linear regression's weighted sum, then squash it into a probability. Estimate `P(instance belongs to the positive class)`:
- `p̂ ≥ 0.5` → predict positive (class 1)
- `p̂ < 0.5` → predict negative (class 0)

A **binary classifier**.

**The model (slide 64):**
1. Compute the weighted sum `t = θᵀx` (same as linear regression).
2. Pass it through the **logistic / sigmoid** function:

```
σ(t) = 1 / (1 + e^(−t))      # outputs a value in (0, 1) for any input
```

**Training (slide 65):**
- We want high `p̂` when `y = 1`, low `p̂` when `y = 0`.
- Cost = **log-loss** (a.k.a. cross-entropy). For one instance it heavily punishes confident-but-wrong predictions.
- ✅ Log-loss is **convex** → GD finds the global minimum.
- ⚠️ **No analytical/closed-form solution** (no normal-equation equivalent) → you *must* use gradient descent.

**Decision boundary (slide 66):** `p̂ = 0.5` exactly when `θᵀx = 0`, which is a **linear** boundary (a point in 1-D, a line in 2-D). Logistic regression is a *linear* classifier — the sigmoid bends the probability, not the boundary.

```python
from sklearn.linear_model import LogisticRegression
X = iris.data[["petal width (cm)", ...]].values
y = (iris.target_names[iris.target] == 'virginica')   # boolean target
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.predict(X_new)
```

---

## Classwork checklist (do these in `04_training_linear_models.ipynb` on Colab)

1. **Slide 22 — Auto MPG, analytical:** model `mpg ~ displacement`. Identify dependent (mpg) vs independent (displacement) var; plot mpg vs displacement; fit `LinearRegression`; report R²; overlay the best-fit line; write the prediction equation from the coefficients; predict at displacement = 250 and **600** (note the extrapolation risk on 600).
2. **Slide 28 — nonlinear:** model `mpg ~ displacement²`; give the regression equation; plot predictions.
3. **Slide 40 — SGD:** same MPG task via `SGDRegressor`; sweep `max_iter` and `eta0`; are predictions sensitive to them? compare to the analytical result.
4. **Slide 62 — LASSO:** iris, model `Petal.Width ~ Petal.Length + Sepal.Length + Sepal.Width`; sweep `alpha`; report coefficients & equation; is it sensitive to alpha? (watch coefficients hit zero).
5. **Slide 68 — Logistic:** iris, classify `virginica` vs not; try feature combinations; use **3-fold cross-validation**; find the combo with highest accuracy.

> Reminder from our running notes: in submission answers, use the instructor's hinted approach (the exact sklearn calls shown on the slides) and don't volunteer alternative methods unless asked.

---

## Exam-trap quick list

- **R² always goes up** with more features (even random) → that's why **Adjusted R²** exists.
- **SSE = Σ(ŷ − y)²** (vs actual), **SSR = Σ(ŷ − ȳ)²** (vs mean). Don't swap them.
- **Feature scaling** matters for **gradient descent**, *not* for the normal equation.
- **Normal equation: O(n²) in features**, linear in instances. GD wins when features are huge or data is streaming.
- **MSE is convex** → GD finds the global min for linear regression. Same is true for **logistic regression's log-loss**.
- **Logistic regression has no closed-form solution** — must use GD. (Linear regression *does* — the normal equation.)
- **Ridge (L2)** shrinks toward zero; **LASSO (L1)** shrinks *to* zero (sparsity/feature selection). LASSO has no analytical solution; Ridge does.
- **High bias = underfit** (high train error); **high variance = overfit** (low train error, big gap to validation).
- Logistic regression's **decision boundary is linear** even though the sigmoid is curved.
