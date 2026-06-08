# CPSC 483 — Day 6 (6/8): Decision Trees

**Géron Ch. 5** · Instructor: Dr. Anand Panangadan
**Status:** Final-only material (post-midterm). Covers DT representation, ID3 (entropy/information gain), CART (Gini), continuous-variable splits, overfitting & pruning, regression trees, and interpretable ML.

> **House note:** Slide deck = Dr. P's "Decision Trees" PPT (80 slides). The interpretable-ML closing block (slides 65–79) is borrowed from Greg Harris (USC). Cross-check any slide claim against Géron when something looks off — standing principle from the SVM C-parameter error.

---

## 1. What a decision tree is (slides 3–11)

A **decision tree** is a predictive model built from a branching series of tests on feature values. You start at the **root**, follow the branch matching each test outcome, and land at a **leaf** that gives the prediction.

- **Internal nodes** = tests on a single attribute (the *splitting attributes*).
- **Branches** = outcomes of that test.
- **Leaves** = class label (classification) or number (regression).

**Worked example — tax cheating.** Given `Refund`, `Marital Status`, `Taxable Income`, predict whether a person cheated. One valid tree:

```
Refund?
├─ Yes → NO
└─ No → Marital Status?
        ├─ Married → NO
        └─ Single/Divorced → Taxable Income?
                             ├─ < 80K → NO
                             └─ > 80K → YES
```

**Expressiveness (slide 5):** decision trees can represent **any Boolean function** of the inputs — each truth-table row becomes one root-to-leaf path. The `A XOR B` tree is the canonical demo.

**Key idea, stated early (slide 8):** *more than one tree can fit the same data.* The tax example has at least two correct trees (root on `Refund` vs. root on `MarSt`). This sets up the central question of *which* tree to prefer.

**Learning = induction (slides 9–10):** "decision tree induction" means finding a tree that agrees with the training data, then applying it (deduction) to new test instances.

---

## 2. Why we want the *smallest* tree (slides 15–17)

- **Naïve approach:** build one path per training example. This just **memorizes** — it doesn't generalize to new data.
- **Occam's Razor:** all else equal, prefer the simplest explanation. → Find the **smallest** tree that classifies the data correctly.
- **The catch:** finding the smallest consistent tree is **NP-hard**. So we don't search for the optimum — we use a **greedy heuristic**.

**Hypothesis-space size (slide 17) — exam-friendly fact:**
With *n* Boolean attributes, the number of distinct decision trees = number of Boolean functions = number of distinct truth tables with 2ⁿ rows =

$$2^{2^n}$$

(For n=6 that's already 2⁶⁴ ≈ 1.8 × 10¹⁹ — illustrates why exhaustive search is hopeless.)

---

## 3. ID3 algorithm — entropy & information gain (slides 12–31)

**Greedy strategy:** at each node, pick the attribute that **maximizes information gain** as the next test. A *perfect* attribute splits the examples into subsets that are each all-positive or all-negative.

### 3.1 Entropy (slide 22)

Entropy measures the **randomness / impurity** of a collection. For a set with *p* positive and *n* negative examples:

$$I(p, n) = -\frac{p}{p+n}\log_2\!\frac{p}{p+n} - \frac{n}{p+n}\log_2\!\frac{n}{p+n}$$

- **Log base 2** → entropy is measured in **bits**.
- Maximum (= 1 bit for two classes) when classes are balanced (50/50).
- Zero when the set is pure (all one class).

**Restaurant dataset (slides 13–14, 23):** 12 examples, 6 positive / 6 negative.
$$I(6,6) = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1 \text{ bit}$$
Need 1 bit of information to classify a randomly picked example.

### 3.2 Information Gain (slide 24)

Information gain = **expected reduction in entropy** from splitting on attribute *A*:

$$Gain(S, A) = I(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|}\, I(S_v)$$

- *Values(A)* = all possible values of *A*.
- *S_v* = subset of *S* where *A* = *v*.
- The summation term is the **weighted average entropy** of the children (weighted by subset size).
- **Higher gain → bigger drop in uncertainty → better attribute.**

### 3.3 Worked gain: Type vs. Patrons (slide 25)

Full set: *p = n = 6*, so *I(S) = 1*.

**Type** (French, Italian, Thai, Burger — each splits 1+/1+ or 2+/2−):
$$Gain(Type) = 1 - \left[\tfrac{2}{12}I(1,1) + \tfrac{2}{12}I(1,1) + \tfrac{4}{12}I(2,2) + \tfrac{4}{12}I(2,2)\right] = 1 - 1 = \mathbf{0}$$
Type is useless — every branch is still 50/50.

**Patrons** (None: 0+/2−, Some: 4+/0−, Full: 2+/4−):
$$Gain(Patrons) = 1 - \left[\tfrac{2}{12}I(0,2) + \tfrac{4}{12}I(4,0) + \tfrac{6}{12}I(2,4)\right] = 1 - 0.459 = \mathbf{0.541}$$
Patrons cuts entropy from 1 → 0.459. **Highest gain → becomes the root.**

> **Why None and Some contribute 0:** *I(0,2)* and *I(4,0)* are both pure subsets → entropy 0. Only the `Full` branch (2+/4−) carries residual uncertainty.

### 3.4 Classwork — second-level split (slides 26–27)

After `Patrons` is the root, the `Full` branch (6 examples, 2+/4−) needs a follow-up test. Compare `Hungry` vs. `Type`:

- **Hungry:** True (2+/2−), False (0+/2−) → Expected info = (4/6)·I(2,2) + (2/6)·I(0,2) = **0.667**
- **Type:** Thai (1,1), French (0,1), Burger (1,1), Italian (0,1) → Expected info = (2/6)·I(1,1) + (1/6)·I(0,1) + (2/6)·I(1,1) + (1/6)·I(0,1) = **0.667**

**Both tie at 0.667.** A genuine tie — either is a valid pick. (The final learned tree on slide 29 happens to use `Hungry` then `Type`.)

### 3.5 The recursive procedure (slide 28)

```
def IDT(D):
    IF all examples in D have the same class c:
        RETURN leaf labeled c
    ELSE IF no attributes left to test:
        RETURN leaf labeled with majority class of D
    ELSE:
        A = "best" attribute (max information gain)
        for each value vᵢ of A:
            Dᵢ = { d in D : d.A == vᵢ }
            subtree tᵢ = IDT(Dᵢ)
        RETURN tree rooted at A with subtrees tᵢ
```

Two base cases (pure node, or out of attributes) + one recursive greedy split. **ID3 was developed in 1986.**

### 3.6 ID3 problems & summary (slides 32–33)

**Advantages:** easy to implement, convertible to rules, human-readable, computationally cheap, fairly robust to noise (probabilistic decisions).

**Limitations:**
- **Not optimal** — greedy, commits early, never reconsiders.
- **Needs discrete attributes** — must discretize continuous ones.
- **Overfits** large/deep trees.
- **Univariate** — tests one attribute at a time, can't natively express `A₁ + A₂ > 20`.
- **Non-incremental** — bad for online learning.
- Same subtrees can repeat across branches.

**Variants that fix these:** C4.5, **CART**, CHAID, MARS, conditional inference trees.

---

## 4. PlayTennis — the classic ID3 exercise (slide 30)

14-day weather dataset; predict `PlayTennis` (Yes/No) from `Outlook`, `Temperature`, `Humidity`, `Wind`. Standard result: **Outlook has the highest information gain** and becomes the root. (Worth drilling by hand — it's the textbook ID3 example and a likely exam template.)

Class balance: 9 Yes / 5 No → *I(9,5) = 0.940 bits* (good practice value to memorize the setup for).

---

## 5. CART algorithm (slides 34–47)

**Classification And Regression Tree.** This is the **default in scikit-learn's `DecisionTree` classes**, so it matters most for the coding side.

### 5.1 Handling continuous variables (slides 35–37)

Two options:
1. **Discretize** into categorical bins — equal-width (open ends) or equal-density (quantiles).
2. **Binary threshold splits** (CART's approach): one branch `X < t`, the other `X ≥ t`.

**Which thresholds matter (slides 36–37):** sort by *X*; you only need to test split points **between examples of different classes** (a *decision stump*). Splits between same-class neighbors never help.

> Example table (slide 37): for X = {1.2-N, 1.5-Y, 1.7-Y, 2.3-Y, 2.5-N, 3.0-N}, only the **N→Y boundary (≈1.35)** and the **Y→N boundary (≈2.4)** are worth evaluating.

### 5.2 Impurity measures (slides 38–40)

A node is **pure/homogeneous** if it holds a single class. Three impurity indices:

| Measure | Formula | Used by | Range |
|---|---|---|---|
| **Entropy** | $-\sum_i p(c_i)\log_2 p(c_i)$ | ID3 | 0 → log₂(n) |
| **Gini** | $1 - \sum_i p(c_i)^2$ | CART | 0 → 1 |
| **Classification error** | $1 - \max_i\{p(c_i)\}$ | — | 0 → 1 |

All three = **0 when pure**, **max when classes are equiprobable**.

**Worked example (slide 39)** — 10 instances, classes A=0.4, B=0.3, C=0.3:
- Entropy = −(0.4·log₂0.4 + 0.3·log₂0.3 + 0.3·log₂0.3) = **1.571**
- Gini = 1 − (0.4² + 0.3² + 0.3²) = 1 − 0.34 = **0.66**

  > ⚠️ **Slide error (slide 39):** the slide writes Gini as `1 − (0.16² + 0.09² + 0.09²) = 0.9582`. That's wrong twice over — it squares the *already-squared* probabilities and the arithmetic doesn't even match. **Correct Gini = 1 − (0.16 + 0.09 + 0.09) = 1 − 0.34 = 0.66.** (Slide 40's text correctly bounds Gini at 0–1, which 0.9582 nearly violates for 3 classes where max Gini = 0.667 — a tell that the number is bogus.) **Use 0.66.**

- Classification error = 1 − 0.4 = **0.60**

**Entropy range subtlety (slide 40):** entropy's max is **log₂(n)** for *n* classes (so it can exceed 1 — e.g., 1.571 above for 3 classes), whereas Gini and classification error are always capped at 1.

### 5.3 The CART split criterion (slide 41)

CART is recursive and **always binary**. At each step it picks the feature *k* and threshold *tₖ* minimizing the **size-weighted Gini of the two children**:

$$J(k, t_k) = \frac{m_{\text{left}}}{m}\,G_{\text{left}} + \frac{m_{\text{right}}}{m}\,G_{\text{right}}$$

where *G* is the impurity of each subset, *m_left/right* are the child sizes, and *m = m_left + m_right*. **Intuition: split to get the purest subsets.** Stop when impurity can't be reduced.

### 5.4 Iris CART tree (slide 42) — read a tree node

```
petal length ≤ 2.45?   gini=0.667  samples=150  value=[50,50,50]
├─ True  → gini=0.0   samples=50   value=[50,0,0]   class=setosa
└─ False → petal width ≤ 1.75?  gini=0.5  samples=100  value=[0,50,50]
           ├─ gini=0.168  samples=54  value=[0,49,5]   class=versicolor
           └─ gini=0.043  samples=46  value=[0,1,45]   class=virginica
```

Each node reports its split test, Gini, sample count, the `value` per-class counts, and the majority `class`. Root Gini = 1 − 3·(1/3)² = 0.667 ✓ (max for 3 balanced classes).

### 5.5 Classwork — Gini splits by hand (slides 43–44)

**Slide 43** — same 6-row table. Compute Gini of each side, then *J*, for **X=1.35** vs **X=2.4**, and decide which split is better.

- **Split at X=1.35:** left = {1.2-N} (pure, G=0), right = {1.5-Y,1.7-Y,2.3-Y,2.5-N,3.0-N} = 3Y/2N → G = 1−(0.6²+0.4²) = 0.48.
  J = (1/6)(0) + (5/6)(0.48) = **0.40**
- **Split at X=2.4:** left = {1.2-N,1.5-Y,1.7-Y,2.3-Y} = 3Y/1N → G = 1−(0.75²+0.25²) = 0.375; right = {2.5-N,3.0-N} (pure, G=0).
  J = (4/6)(0.375) + (2/6)(0) = **0.25**
- **0.25 < 0.40 → split at X=2.4 is better** (lower weighted impurity).

**Slide 44** — iris, versicolor + virginica only, split on `Petal length` at 5.0. Same procedure: Gini left, Gini right, then *J(Petal length, 5.0)*. (Drill this — it's the multiclass→binary-filter pattern, and `fetch_openml`/`load_iris` label-type gotchas apply if you code it: compare with the right label dtype.)

### 5.6 scikit-learn code (slides 45–47)

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", ...]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_iris, y_iris)
```

**Classwork (slide 46):** load `autompg.csv` (Canvas), use `DecisionTreeClassifier` to predict `Cylinders` from the other features, **drop the last column (model name)**, and visualize with **graphviz** (textbook code). Textbook notebook: **`05_decision_trees.ipynb`**.

> **Auto MPG reminder (house log):** watch the 398 vs. 392 row count (6 rows have missing `horsepower`) and state your row count explicitly — it's grading-relevant.

---

## 6. Overfitting & pruning (slides 48–57)

### 6.1 Why trees overfit (slide 49)

- Running DT learning **to completion always overfits** — a fully grown tree has **no learning bias**, so **training error → 0** (unless two identical-feature examples have different labels).
- A decision tree is a **nonparametric model** — the number of parameters is **determined by the data**, not fixed in advance.
- As the tree grows, leaves match single examples; **bias drops but variance rises**, so new-data accuracy degrades.
- **Training error stops predicting test error** past the overfitting point (the classic train↓ / test↑ divergence curve).

### 6.2 Pre-pruning = early stopping (slides 50–51)

Stop growing **before** the tree is full. Stopping conditions:
- All instances same class, or all attribute values identical (if labels then differ → noise → majority vote).
- More restrictive: fixed max depth, node-size threshold, or gain/impurity-improvement below a threshold.

**sklearn regularization hyperparameters (slide 51):**

| Hyperparameter | Meaning |
|---|---|
| `max_features` | Max features evaluated per split |
| `max_leaf_nodes` | Max number of leaves |
| `min_samples_split` | Min samples a node needs before it can split |
| `min_samples_leaf` | Min samples required in a created leaf |
| `min_weight_fraction_leaf` | Like `min_samples_leaf` but as a fraction of total weighted instances |

> **Regularization rule (slide 51, exam-likely):** **increasing `min_*`** OR **decreasing `max_*`** regularizes the model (simpler tree, less overfitting).

### 6.3 Post-pruning (slides 52, 56–57)

Grow the **full** tree, then prune back:
1. Split into **training + validation** sets (validation ≠ test).
2. Grow the full tree on training.
3. Prune while it helps: replace a subtree's root with a leaf labeled by the subtree's **majority class** (or the class that most boosts **validation accuracy** — greedy).
4. Evaluate each subtree's pruning impact.

**Post- vs. pre-pruning:** post-pruning generally wins — it's hard to know in advance when to stop growing.

### 6.4 Trees → rules → simplification (slides 53–55)

- Each **root-to-leaf path** = one classification rule. The full rule set is **mutually exclusive and exhaustive** and carries **as much information as the tree**.
- Rules can be **simplified** by deleting preconditions that don't hurt accuracy. Example (slide 54): `(Refund=No) ∧ (Status=Married) → No` simplifies to just `(Status=Married) → No` (every Married row is No regardless of Refund).
- **Classwork (slide 55):** write the rule for the **longest path** in the restaurant tree (Patrons=Full ∧ Hungry=No ∧ Type=Thai ∧ Fri/Sat=…) and list all valid simplifications.

### 6.5 C4.5's pruning strategy (slides 56–57)

The full pipeline — build tree → convert to rules → prune each rule **independently** (delete preconditions that improve accuracy) → **sort** simplified rules by priority (they may no longer be mutually exclusive, so >1 can match) → classify with the sorted set. This is the strategy of **C4.5**, one of the most successful DT algorithms.

---

## 7. Regression trees (slides 58–64)

**Idea:** instead of a class, each **leaf predicts a number** — the **average target value** of the training samples that land in that leaf.

### 7.1 The regression split criterion (slide 59)

Same weighted form as CART classification, but minimize **MSE** instead of Gini:

$$J(k, t_k) = \frac{m_{\text{left}}}{m}\,\text{MSE}_{\text{left}} + \frac{m_{\text{right}}}{m}\,\text{MSE}_{\text{right}}$$

where for each node:

$$\text{MSE}_{\text{node}} = \sum_{i \in \text{node}} (\hat{y}_{\text{node}} - y^{(i)})^2 \Big/ m_{\text{node}}, \qquad \hat{y}_{\text{node}} = \frac{1}{m_{\text{node}}}\sum_{i \in \text{node}} y^{(i)}$$

The prediction $\hat{y}_{\text{node}}$ is just the **mean target** in the node. Deeper tree → finer **piecewise-constant** (staircase) approximation.

### 7.2 Geometry (slides 60, 62)

- The X-space is partitioned into **disjoint axis-parallel regions** — every split tests one variable against one threshold, so all boundaries are parallel to the axes.
- **Linear regression vs. regression tree:** linear regression fits a smooth tilted plane; a regression tree fits a **stepped surface**. The tree is **nonparametric**.

### 7.3 Overfitting in regression trees (slide 61)

Unrestricted → the staircase chases every noisy point. Setting e.g. `min_samples_leaf=10` smooths it into sensible steps. Same `min_*` / `max_*` regularization as classification trees.

### 7.4 Pros & cons (slide 63)

**Pros:** easy to see which variables matter; handles missing data (average the subtree leaves); works even when the true surface isn't smooth; fast to train.
**Cons:** imprecise predictions; **can't extrapolate beyond the training range** (a hard ceiling/floor — important caveat); overfits noisy data.

> Textbook notebook again: **`05_decision_trees.ipynb`** (slide 64).

---

## 8. Interpretable ML (slides 65–79, USC/Greg Harris block)

Why care about interpretability when interpretable models often trade away accuracy vs. black boxes?

- **Informativeness (slide 68):** ML as *knowledge discovery*, not just automation — sometimes the prediction itself is worthless and the **insight** is the product (e.g., where to place grocery items).
- **Trust (slides 69–72):** people distrust opaque models — *"Trust in a system is developed not only by the quality of its results, but also by clear description of how they were derived"* (Swartout 1983). Interpretability helps spot overfitting. The **"Stupid Data Miner Tricks" S&P 500** example (Leinweber 2007 — Bangladesh butter production predicting the S&P at R²=0.99) is the cautionary tale: spurious correlation + multiple-comparisons → false discovery.
- **Causality (slide 73):** ML finds **association, not causation**. Interpretable models *suggest* relationships to then test experimentally. Causal relationships let you **alter** (not just predict) and **cure** (not just diagnose).
- **Transferability (slide 74):** models are brittle in new environments; humans judge transfer better. The **tank example** (a CNN that learned to detect *sunny vs. cloudy* instead of tanks) and the **pneumonia/asthma** example (model learned asthmatics had *lower* death risk — because they got more aggressive care, a confound that would kill if deployed naively).

**Taxonomy of interpretability:**
- **Comprehensibility (slide 76):** can you simulate it in your head? **Comprehensible:** decision trees, classification rules, linear models. **Incomprehensible:** deep nets, kernelized SVMs, random forests, ensembles.
- **Decompositionality (slide 77):** can the model break into sensible parts? Needs meaningful (not anonymous/engineered) features.
- **Post-hoc interpretability (slides 78–79):** keep accuracy, add explanation after the fact — case-based (*k*-NN in medicine), LDA topics, **class saliency maps** for CNN image classification (highlight the discriminative pixels).

---

## Exam-trap checklist

1. **Information gain = parent entropy − weighted-average child entropy.** Don't forget the size weights |Sᵥ|/|S|.
2. **Entropy uses log₂** (answer in bits). Entropy max = **log₂(n)**, so it can exceed 1; **Gini and classification error cap at 1.**
3. **Pure subset → I = 0 and Gini = 0.** Those branches drop out of the weighted sum.
4. **Gini = 1 − Σp²** (not 1 − Σ(p²)² — the slide-39 trap). For 10 instances split 0.4/0.3/0.3, Gini = **0.66**, not 0.9582.
5. **CART = binary splits + Gini; ID3 = multiway splits + entropy.** CART is sklearn's default.
6. **Continuous splits:** only test thresholds **between different-class neighbors** after sorting.
7. **Regularization direction:** **↑ `min_*`** or **↓ `max_*`** = simpler = less overfit.
8. **Pre-pruning = stop early; post-pruning = grow full then cut back.** Post usually better.
9. **Regression-tree leaf = mean of its samples; split minimizes weighted MSE.** Can't extrapolate beyond training range.
10. **A tree has no learning bias → training error 0 → guaranteed overfit if grown fully.** Nonparametric.
11. **Tie-breaking is real** — restaurant `Hungry` vs. `Type` both give 0.667; either is correct.
12. **Multiple trees fit the same data** (slide 8). Don't assume uniqueness.

## Self-check (predict before peeking — exam is AI-free)

1. Dataset with 9 Yes / 5 No. What's the entropy in bits? *(≈0.940)*
2. An attribute splits 14 examples into {9+/0−} and {0+/5−}. Information gain? *(= 0.940 − 0 = 0.940 — a perfect split)*
3. Three classes, perfectly balanced. Gini? Entropy? *(Gini = 1−3(1/3)² = 0.667; Entropy = log₂3 ≈ 1.585)*
4. Why does CART only consider thresholds between different-class points? *(Same-class boundaries can't reduce impurity — no gain.)*
5. You set `min_samples_leaf` from 1 to 20. More or less overfitting? *(Less — simpler tree.)*
6. A regression tree is asked to predict for x far above any training x. What happens? *(It returns the nearest leaf's mean — flat, no extrapolation.)*
7. Why does training error mislead you for a fully-grown tree? *(It's ~0 by construction; says nothing about generalization.)*
8. Name three comprehensible and three incomprehensible model types. *(Comp: trees, rules, linear. Incomp: deep nets, kernel SVM, random forests/ensembles.)*

---

*Cross-check slides against Géron when they conflict (SVM C-parameter; slide-39 Gini). Optimize for unaided fluency — the final is AI-free. This is now final-exam material (Ch 5).*
