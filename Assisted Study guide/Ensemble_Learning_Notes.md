# CPSC 483 — Ensemble Learning — Lecture Notes

**Course:** CPSC 483-02, Introduction to Machine Learning (Summer 2026)
**Instructor:** Dr. Anand Panangadan
**Textbook ref:** Géron, *Hands-On Machine Learning* — Ensemble Learning & Random Forests chapter
**Deck:** `ML_Ensemble_Learning.pdf` (43 slides) · notebook `06_ensemble_learning_and_random_forests.ipynb` (`handson-mlp` repo)
**Cross-check status:** ✅ Clean — no computational errors found (details in the cross-check log at the end)

---

## 0. The one idea that ties the whole lecture together

Every method in this deck is an answer to a single question: **"A single model isn't good enough — how do I combine many models into one that's better?"** The differences between them come down to *how* they generate the members and *how* they combine the votes.

The cleanest lens is the **bias–variance decomposition**:

- **Bagging / Random Forests / Extra-Trees → attack VARIANCE.** Build many *low-bias, high-variance* models (deep trees) in **parallel and independently**, then average. Averaging cancels out the random wiggles, leaving the signal. Bias stays roughly the same; variance drops.
- **Boosting → attacks BIAS (and variance too).** Build many *high-bias, low-variance* models (shallow stumps) **sequentially**, each one focused on the mistakes of the last. The committee gradually bends toward the hard cases, driving bias down.
- **Voting / Stacking → combine *different kinds* of models.** Voting uses a fixed rule (majority / average); stacking *learns* the combining rule.

| Family | Members built… | Member type | Primarily reduces | Analogy |
|---|---|---|---|---|
| Voting | in parallel, different algorithms | mixed | variance | a panel of different specialists votes |
| Bagging | in parallel, same algorithm, resampled data | high-variance | **variance** | many copies of one expert, each seeing a different sample |
| Random Forest | bagging + random features per split | high-variance | **variance** (more than bagging) | force each expert to ignore some clues so they stop agreeing |
| Boosting | **sequentially**, reweighting mistakes | high-bias (weak) | **bias** | a student re-studying only the problems they got wrong |
| Stacking | base layer + a learned blender | mixed | both | a manager who *learns* which specialist to trust when |

Keep this table in your head. Almost every exam question on this chapter is testing whether you know *which column a method lives in*.

---

## Slide-order notes

### Slides 1–2 · Roadmap
Five techniques, in this order: **voting classifiers → bagging → random forests → boosting → stacking.** Roughly: voting and bagging are the foundations, random forests are the headline bagging method, boosting is the other major family, and stacking is the "learned aggregation" capstone.

---

### Slides 3–4 · Voting classifiers — the core idea (hard voting)
Train **several different classifiers on the same data**, then for a new instance let them vote; the class with the most votes wins. This majority-vote scheme is a **hard voting classifier**.

**Why it works (the surprising part):** even if each classifier is only mediocre, their *errors are partly independent*, so they don't all make the same mistake on the same instance. When you take a majority vote, the independent errors tend to cancel and the correct answer survives. This is the "wisdom of the crowd" — a committee of so-so members can beat any individual member, **as long as the members are diverse and better than random.** (Slides 7–8 make this rigorous.)

The slide figure: several distinct classifiers (e.g., LR, RF, SVM) each draw their own decision boundary; the ensemble's prediction is the majority across them.

### Slide 5 · Soft voting
If every classifier can output a **class probability** (not just a label), average those probabilities and pick the class with the highest **average probability** — this is **soft voting**.

**Why soft usually beats hard:** soft voting gives **more weight to confident votes.** A classifier that says "90% class A" counts more strongly than one that barely says "51% class A." Hard voting throws that confidence information away (a 51% vote and a 99% vote count identically). Soft voting keeps it, so a few highly-confident correct votes can outweigh several lukewarm wrong ones.

> ⚠️ Practical gotcha (not on the slide but exam-friendly): for soft voting in scikit-learn you must set `voting='soft'`, **and every estimator must expose `predict_proba`.** `SVC` does **not** by default — you need `SVC(probability=True)`, which is slower. The slide-6 code uses the default (`voting='hard'`).

### Slide 6 · Voting in scikit-learn
```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

voting_clf = VotingClassifier(
    estimators=[('lr', LogisticRegression()),
                ('rf', RandomForestClassifier()),
                ('svc', SVC())]
)
voting_clf.fit(X_train, y_train)
```
Three *different* algorithms (diverse members) on the *same* `make_moons` data. Default is hard voting. Note this is diversity-by-**algorithm**; bagging (next) gets diversity by **data**.

---

### Slide 7 · Weak vs. strong learners
- **Weak learner:** only slightly better than random guessing (e.g., ~51% on a balanced binary problem).
- **Strong learner:** high accuracy.
- **The headline theorem:** an ensemble of weak learners *can be* a strong learner — **provided (1) there are enough of them and (2) they are sufficiently diverse.**

Those two conditions are the recurring theme. "Enough" gives the law-of-large-numbers averaging effect; "diverse" is what makes their errors independent enough to cancel. If all members make the *same* mistakes, averaging buys you nothing.

### Slide 8 · The biased-coin intuition (memorize this)
A coin lands heads 51% of the time. Which side is biased?
- **One toss:** you're wrong ~**49%** of the time — barely better than a coin flip.
- **1000 tosses, majority rule:** wrong only ~**25%** of the time.

I verified the second number: with $p=0.51$ over $N=1000$ flips, $P(\text{majority is the wrong side}) \approx 0.25$ by the normal approximation. The point: **each toss is a "weak learner."** Pooling 1000 of them turns a 51%-edge into a ~75%-correct decision. That is *exactly* what an ensemble does — it amplifies a tiny per-model edge into a large ensemble edge, **as long as the tosses are independent.** Correlation between members is the silent killer that breaks this (which is why random forests work so hard to decorrelate trees — slide 21).

---

### Slide 9 · Bagging — getting diversity from *data* instead of *algorithms*
Two ways to get a diverse set of classifiers:
1. **Different algorithms** on the same data (that's voting, slides 3–6).
2. **One algorithm** trained on **different random subsets** of the data → **bagging = bootstrap aggregating.**

Bagging is attractive because you only need *one* learning algorithm; the diversity comes for free from the resampling.

### Slide 10 · Bootstrap sampling (the engine of bagging)
Given a dataset $D$ of $N$ examples, build subset $D_i$ by **drawing $n$ examples at random *with replacement*.** Because it's with replacement, some examples repeat and others are missed.

> Example: $D=\{1,2,3\}$ → $D_1=\{1,1,2\}$, $D_2=\{1,2,3\}$, $D_3=\{1,3,3\}$.

- **Empirical bootstrap:** resample from the observed data (what we use here).
- **Parametric bootstrap:** sample from a fitted distribution (e.g., a Normal).

**Why with replacement matters:** it's what makes each $D_i$ genuinely *different* (different multiset of points, different duplicates). Without replacement you'd just be reshuffling the same set. Think of each bootstrap sample as a slightly different "parallel universe" version of your data — train one model per universe and the models disagree just enough to be useful when averaged.

### Slide 11 · Classwork #1 — generate 3 bootstrap samples
```python
import random
n = 5
data_list = list(range(n))     # [0, 1, 2, 3, 4]
print(data_list)
# run the next two lines 3 times:
selected_items = random.choices(data_list, k=n)   # WITH replacement
print(selected_items)
```
*What you should observe:* each run gives a length-5 sample with some values repeated and some of `{0,1,2,3,4}` missing. The missing ones are your out-of-bag instances (foreshadowing slide 15). `random.choices` = with replacement; contrast with `random.sample` = without. **(Your turn — see Classwork tracker below.)**

---

### Slide 12 · Bagging — prediction & the variance argument
After training, the ensemble predicts a new instance by **aggregating** all members:
- **Classification → majority vote (the mode).**
- **Regression → average (the mean).**

**The key claim, and why:** the ensemble has **similar bias but lower variance** than a single predictor. Each tree trained on a bootstrap sample is a high-variance estimate of the same underlying function — same expected value (similar bias), but each one wiggles differently around it. Averaging many such estimates keeps the expected value (bias unchanged) while the variance of the average shrinks. In the idealized independent case, averaging $K$ estimators divides variance by $K$; in reality the trees are correlated, so the reduction is smaller — but still real.

> 🧠 Connect to slide 8: "majority vote of high-variance trees" *is* the biased-coin trick. The bias of the average = the bias of one tree; the variance of the average ≪ variance of one tree.

### Slide 13 · Example — single tree vs. bagging ensemble (moons)
The figure (Géron's classic): **left**, a single decision tree carves a jagged, chunky, staircase boundary that clearly overfits noise; **right**, a 500-tree bagging ensemble produces a **much smoother** boundary that generalizes better. This is variance reduction made visible — the wiggly idiosyncrasies of individual trees average out.

### Slide 14 · Example — ozone vs. temperature (Rousseeuw & Leroy, 1986)
100 bootstrap samples → fit a smoother to each → predict across the data range → **average the 100 smoothers into one bagged predictor (the red line).** The bagged curve is **smoother (less overfitting)** than any individual smoother. Same lesson as slide 13, but for regression: averaging tames variance.

---

### Slide 15 · Out-of-Bag (OOB) evaluation — free validation
Because bootstrap sampling is *with replacement*, some training instances are never drawn for a given predictor. **Approximately 37% are left out** of each bootstrap sample — these are that predictor's **out-of-bag (OOB) instances.**

**Where 37% comes from (likely exam item):** the chance a specific instance is *not* picked in one draw is $(1-\tfrac1n)$; over $n$ independent draws it's
$$\left(1-\frac{1}{n}\right)^n \xrightarrow[n\to\infty]{} e^{-1} \approx 0.368 \approx 37\%.$$
So ~63% of instances are "in-bag" and ~37% are OOB. (I checked: $n{=}100 \to 0.366$, $n{=}1000 \to 0.368$.)

**Why this is a free lunch:** each instance is OOB for ~37% of the trees, so you can evaluate it using *only the trees that never saw it during training* — a legitimate held-out estimate **without sacrificing any data to a validation set.** Different predictors have different OOB sets. You can use the aggregated OOB error to **set hyperparameters** and to decide when to stop adding trees (slide 23).

### Slide 16 · Classwork #2 — same bootstrap code, now identify the OOB samples
Re-run slide 11's code; for each sample, the OOB set is `set(data_list) - set(selected_items)`. You'll almost always find at least one OOB element in a size-5 sample (the expected count is $5 \times 0.37 \approx 1.85$). **(Your turn.)**

---

### Slide 17 · Random subspaces & random patches
Instead of sampling **rows (instances)**, sample **columns (features)**:
- **Random subspaces method:** randomize **features** only (keep all rows). Great when you have **many features**.
- **Random patches method:** randomize **both rows and columns.**

**Why feature sampling helps:** it **decorrelates** the models. If one feature is overwhelmingly predictive, every model trained on all features will lean on it and they'll all look alike — averaging clones buys nothing (back to the "diversity" condition from slide 7). Hiding random subsets of features forces models to find *other* signal, so they disagree more, so averaging helps more. This is the bridge to random forests, which bake feature-sampling in at every split.

---

## RANDOM FORESTS (slides 18–29)

### Slide 19 · Why plain decision trees need help
Trees are **interpretable and fast** (greedy ID3/CART), **but**:
- To capture a complex boundary you need a **large** tree.
- Each node makes only **axis-aligned splits** (perpendicular to a feature axis) — so curved/diagonal boundaries get approximated by a blocky staircase.
- **Large trees → high variance → overfitting.**
- In practice, lone trees **often underperform** other methods.

This is the setup for the fix: keep the tree's flexibility, kill its variance by ensembling. (Note: a forest *cannot* fix axis-aligned splits — every member is still axis-aligned — but averaging many of them produces an *effectively* smoother boundary, as the slide-13 picture shows.)

### Slide 20 · The Random Forest algorithm
An ensemble **specifically designed for decision trees.** Training:
1. Draw **$K$ bootstrap samples** $D_1, \dots, D_K$ from $D$.
2. Grow a tree from each sample. **At every node**, instead of considering all features:
   - **select $m$ attributes at random**, pick the **best** of those (best by **Gini** for classification, **MSE** for regression), choose a split point, and split.
3. Return the **ensemble of trees.**

**Prediction:** majority vote (classification) or average (regression) across all trees. The "(used in CART)" note just means the base learners are CART trees.

### Slide 21 · Two sources of randomness (this is THE random-forest exam point)
1. **Bagging** — each tree sees a different **bootstrap sample of rows.**
2. **Random attributes** — at **each node**, the best split is chosen from a **random subset of $m$ features**, not all of them.

**Why both?** Bagging alone leaves trees fairly correlated (they keep splitting on the same dominant feature first). Adding per-node feature randomness **decorrelates** the trees further, which is precisely what makes the average tighter (slide 8's independence requirement again). **More decorrelation → more variance reduction.** This is why a random forest typically beats plain bagged trees: it trades a tiny bit of individual-tree accuracy for much lower correlation between trees.

> ⚠️ Easy to confuse: feature randomness is applied **at each split node**, *re-drawn every node* — not once per tree. (Random subspaces from slide 17 picks features once per tree; random forests re-pick at every node.)

### Slides 22–23 · Tuning random forests
Hyperparameters:
1. **$m$ = number of predictors sampled per split.**
2. **Number of trees** in the ensemble.
3. **Minimum leaf size** (trees *can* be full, but that's costly).

**Failure mode worth knowing:** when there are **many features but few relevant ones**, a random $m$-subset rarely contains a relevant feature, so most trees end up weak → the forest underperforms. (Mitigation: larger $m$, or feature selection upstream.)

**Rule-of-thumb defaults for $m$ (Breiman):**
$$m = \sqrt{N}\ \text{(classification)}, \qquad m = \frac{N}{3}\ \text{(regression)},$$
where $N$ = total number of features. **Better than rules of thumb:** tune $m$ and the tree count using **OOB error** — it's data-specific, and OOB lets you fold training + cross-validation into one pass. **Stop adding trees once the OOB error stabilizes.**

> ⚠️ **Library-vs-theory nuance (not a slide error — but worth knowing):** scikit-learn's `RandomForestClassifier` defaults to `max_features='sqrt'` (= $\sqrt{N}$ ✓ matches the slide). But `RandomForestRegressor` defaults to `max_features=1.0` (**all** features), **not** $N/3$. So $N/3$ is Breiman's classic recommendation, while sklearn's regressor default differs. If an exam asks "what's the rule of thumb," answer $\sqrt N$ / $N/3$; if it asks "what does sklearn do by default for regression," it's all features.

### Slide 24 · Random forests in scikit-learn
```python
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```
`n_estimators` = number of trees; `max_leaf_nodes` = regularizes tree size. There's a parallel `RandomForestRegressor`.

### Slide 25 · Classwork #3 — autompg random forest
- Load `autompg.csv`.
- `RandomForestClassifier(n_estimators=100)` to classify **Cylinders** from the other features.
- **Drop the last column (car/model name)** — it's a high-cardinality text identifier, useless as a feature.
- Visualize one of the trees.
> 🔴 **Slide 25 corrections (confirmed in class).** Two things on this slide turned out differently than the slide text implies:
> 1. **"Visualize the decision tree" was struck out by the instructor in lecture** — and rightly so. A random forest's prediction is a *majority vote across all 100 trees*; no single tree is the model, and each is deliberately made unrepresentative (bootstrap rows + random per-node feature subsets + full depth). Plotting one tree shows one resampled slice's quirks, not the forest's reasoning — exactly the slide-29 black-box property. **The forest-native replacement is `feature_importances_` (slide 27) plus an honest per-class evaluation.**
> 2. **Canvas `autompg.csv` profile (pinned down from the real file):** comma-delimited, quoted fields; last column is **`car_name`**; **`origin` is already numeric (1/2/3)** — *no encoding needed*; `horsepower` carries **6 `NaN`s**, stored as real NaN (**not** `'?'` strings, so no string-coercion step). Drop the name column + `dropna()` and the 7 features are model-ready.
>
> *Result on the real file:* 392 samples × 7 features; with an 80/20 stratified split, test accuracy ≈ **0.975**, OOB ≈ **0.965** (the slide-15 free-validation cross-check agreeing with the held-out number). The class is wildly imbalanced (4-cyl: 199, 8-cyl: 103, 6-cyl: 83, but 3-cyl: 4 and 5-cyl: 3), so **macro-F1 ≈ 0.59 vs weighted-F1 ≈ 0.97** — excellent on common classes, blind on the rare ones. Report per-class metrics, not bare accuracy. **(Your turn — see the delivered notebook for the worked version.)**

### Slide 26 · Textbook code
Open `06_ensemble_learning_and_random_forests.ipynb` (Géron) on Colab as the canonical style reference. **Confirmed correct:** in the `handson-mlp` repo (the Scikit-Learn & PyTorch edition your course uses), ensemble learning genuinely is notebook **06**. *(My earlier "06 vs 07" flag was a false alarm — I'd checked the older `handson-ml3`/Keras repo, where it's 07. Standing lesson: cross-check notebook references against **`handson-mlp`**, not `ml3`.)* Géron's idioms from this notebook: forest named **`rnd_clf`**, instantiated with **`n_jobs=-1, random_state=42`**; feature importance via `for score, name in zip(rnd_clf.feature_importances_, columns): print(round(score, 2), name)`.

### Slide 27 · Feature importance (a free side-benefit)
A random forest gives you a **relative importance score per feature**: how much the nodes that split on that feature **reduce impurity (Gini/MSE)**, **averaged across all trees**, **weighted** by the number of training samples reaching each node.

**Why you get this for free:** you're already measuring impurity reduction at every split to *grow* the trees — feature importance just totals those reductions per feature. It's a cheap, built-in form of **dimensionality reduction / feature ranking** (slide 29 lists it as an advantage).

> ⚠️ **Two caveats — flagged as OUTSIDE the deck (general ML / sklearn knowledge, not examinable from these slides):**
> - Impurity-based importance is **biased toward high-cardinality / continuous features.**
> - **One-hot encoding *fragments* a feature's importance across its dummy columns.** On auto-mpg, encoding `origin` splits its single ~0.037 importance into `origin_1` ≈ 0.023, `origin_2` ≈ 0.010, `origin_3` ≈ 0.004 — same total, harder to read. This connects to the **slide-22 failure mode** (many sparse columns, each rarely chosen under per-node feature subsampling), which is why one-hot can *hurt* tree ensembles on high-cardinality categoricals. The slide-22 link is fair; the fragmentation claim itself is mine — verify independently, don't assume it's exam material.

### Slide 28 · Extremely Randomized Trees (Extra-Trees)
Add **even more randomness**: instead of *searching* for the best split point on each candidate feature, **pick the split point at random** too, then keep the best among those random splits.
- **Much faster** to train (no expensive threshold search).
- **Lower variance at the cost of higher bias** — the extra randomness decorrelates trees even more (↓ variance) but each tree is individually weaker (↑ bias).

**The pattern to internalize:** Bagging → RandomForest → Extra-Trees is a **ladder of increasing randomness.** Each rung adds decorrelation (cuts variance) and costs a little individual-model quality (adds bias). Where you stop on the ladder is the bias–variance dial.

### Slide 29 · Random forest pros & cons
**Pros:** lower variance than a single tree; handles large, high-dimensional data; **trees train in parallel**; copes with missing data; **OOB samples = free test set**; **feature importance** for dimensionality reduction.
**Cons:** a **black box** — you can trace the logic of one tree, but not of hundreds. *"The averaged model is no longer easily interpretable — one can no longer trace the 'logic' of an output through a series of decisions."* This is the classic **accuracy-vs-interpretability trade-off**: you bought accuracy with the very averaging that destroyed the single-tree transparency.

---

## BOOSTING (slides 30–40)

### Slide 31 · Boosting — the big shift
**Train predictors *sequentially*, each one trying to correct its predecessor.** Two variants in this course: **AdaBoost** (Adaptive Boosting) and **Gradient Boosting.**

> 🧠 **The crucial contrast with bagging.** Bagging = **parallel, independent, reduces variance.** Boosting = **sequential, dependent, reduces bias** (each learner is built *in response to* the last one's errors). You **cannot parallelize** boosting across iterations the way you can bagging — that's a frequent exam distinction.

### Slides 32 & 34 · AdaBoost — the algorithm
1. Train a **base classifier** (e.g., a small tree); predict on the training set.
2. **Increase the relative weight of the *misclassified* instances.**
3. Train a **second classifier using the updated weights** — info about each sample's **"hardness"** is fed in so **later trees focus on the hard-to-classify examples.**
4. Predict again, update weights again.
5. **Iterate.**

The analogy: a student who, after each practice test, **spends more study time on exactly the questions they got wrong.** Over rounds, the committee's collective attention concentrates on the genuinely hard region of the data — which is how boosting drives **bias** down.

### Slide 33 · AdaBoost, visually
Three stacked rows: **bottom** = the training data with **instance weights** (dot sizes); **middle** = the predictor at that iteration; **top** = the resulting fit (red curve). The diagonal arrows show predictions flowing back to **reweight** the data for the next round. Watch the bottom row left-to-right: **misclassified points grow larger** (heavier weight), and each new model bends toward them.

### Slide 35 · Decision stump (AdaBoost's default base learner)
A **decision stump** = a tree with `max_depth=1`: **one decision node, two leaves**, predicting from **a single feature.** It's a textbook **weak learner** — and it's the **default base estimator** in scikit-learn's `AdaBoostClassifier`.

**Why deliberately weak?** Boosting's whole job is to *manufacture* a strong learner from weak ones (slide 7). Starting from high-bias stumps gives boosting room to reduce bias round by round. Start from strong learners and you'd overfit fast and lose the gradual error-correction dynamic.

### Slide 36 · AdaBoost — the math (this slide's equations were garbled in the text; here they are, verified against Géron)

Setup: predictor $j$ has weight $\alpha_j$; instance $i$ has weight $w^{(i)}$, **initialized to $1/m$** (so they sum to 1).

**1. Weighted error rate of predictor $j$:**
$$r_j = \frac{\displaystyle\sum_{\substack{i=1\\ \hat{y}_j^{(i)} \neq y^{(i)}}}^{m} w^{(i)}}{\displaystyle\sum_{i=1}^{m} w^{(i)}}$$
i.e., the total weight sitting on the instances predictor $j$ got wrong. *(The slide writes only the numerator $\sum_{\hat y \ne y} w^{(i)}$ — that's correct **because** the weights are kept normalized to sum to 1, making the denominator $=1$. Not an error.)*

**2. Predictor weight (how much say it gets in the final vote):**
$$\alpha_j = \eta \, \log\frac{1 - r_j}{r_j}$$
where $\eta$ = **learning rate** (default 1). Read it:
- low error $r_j \to \alpha_j$ large and **positive** (trusted a lot);
- error ≈ 50% (random) $\to \alpha_j \approx 0$ (ignored);
- error > 50% (worse than random) $\to \alpha_j$ **negative** (its vote is *flipped*).

**3. Instance weight update** (for $i = 1,\dots,m$):
$$w^{(i)} \leftarrow \begin{cases} w^{(i)} & \text{if } \hat{y}_j^{(i)} = y^{(i)} \quad(\text{correct: unchanged})\\[4pt] w^{(i)}\exp(\alpha_j) & \text{if } \hat{y}_j^{(i)} \neq y^{(i)} \quad(\text{wrong: boosted})\end{cases}$$
Then **renormalize** by dividing by $\sum_i w^{(i)}$. A misclassified point gets multiplied by $e^{\alpha_j}$ — so the more accurate the current predictor (bigger $\alpha_j$), the *harder* it pushes the next predictor toward the points it still missed.

**4. Train the next predictor on the updated weights, and repeat.**

> 🧠 Why these exact forms? $\alpha_j$ is monotonic in accuracy, so good predictors dominate the vote. The $\exp(\alpha_j)$ update means *confident-but-still-wrong* regions get aggressively re-emphasized. Together they implement "trust the accurate, obsess over the unsolved."

### Slide 37 · AdaBoost prediction
Run all predictors, **weight each vote by its $\alpha_j$**, and output the class with the **majority of weighted votes**:
$$\hat{y}(\mathbf{x}) = \underset{k}{\arg\max} \sum_{\substack{j=1\\ \hat{y}_j(\mathbf{x})=k}}^{N} \alpha_j.$$
(Not a plain majority — a **weighted** majority, where more accurate predictors count more.)

### Slide 38 · AdaBoost on moons
The figure shows successive AdaBoost iterations carving an increasingly refined boundary on the moons data as later stumps target the still-misclassified points.

### Slide 39 · AdaBoost in scikit-learn
```python
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),   # decision stump
    n_estimators=30,
    learning_rate=0.5)
ada_clf.fit(X_train, y_train)
```
`learning_rate` is the $\eta$ from slide 36; it scales every $\alpha_j$. **There's a trade-off with `n_estimators`:** a *smaller* learning rate shrinks each predictor's contribution, so you typically need *more* estimators to compensate (and it tends to generalize better — the boosting analogue of a small step size).

### Slide 40 · Classwork #4 — AdaBoost on Iris
- Predict **Species** from **all 4 features**; 80/20 train-test split.
- **Vary the learning rate** — does accuracy change?
*What to expect / how to think about it:* Iris is nearly linearly separable, so accuracy is already high and may look insensitive to small learning-rate changes — but very small rates with few estimators can underfit, and very large rates can overshoot/overfit. Sweep a few values (e.g., 0.1, 0.5, 1.0, 2.0) and watch test accuracy *and* train accuracy to see the effect. **(Your turn.)**

---

## STACKING (slides 41–42)

### Slide 42 · Stacking — learn the aggregation
**Main idea:** instead of a *fixed* aggregation rule (hard voting, averaging), **train a model to do the final aggregation.**

The figure (Géron): a **new instance** is fed to several **base predictors** (the "Predict" layer), which output predictions (e.g., 3.1, 2.7, 2.9). Those predictions become the **inputs to a blender / meta-learner** (the "Blending" model), which outputs the **final prediction** (3.0).

**Why this can beat voting:** voting *assumes* the best way to combine is "majority" or "mean." Stacking instead **learns** the best combination from data — including learning that some base models are more reliable on certain inputs, or that two of them are redundant. The blender is a **manager who has learned which specialist to trust when**, rather than blindly averaging everyone.

> ⚠️ Critical implementation note (so the blender doesn't cheat): the blender must be trained on base-model predictions for data the base models **did not train on** — typically via a hold-out set or out-of-fold predictions. If you train the blender on in-sample base predictions, the base models look artificially perfect and the blender overfits. (scikit-learn's `StackingClassifier`/`StackingRegressor` handle this with internal cross-validation.)

---

## 🎯 Exam-trap checklist

1. **Bagging/RF reduce VARIANCE; Boosting reduces BIAS.** If you remember one thing, remember this column assignment.
2. **Bagging = parallel & independent; Boosting = sequential & dependent.** Boosting can't be parallelized across iterations.
3. **OOB ≈ 37%** comes from $(1-1/n)^n \to e^{-1} \approx 0.368$. In-bag ≈ 63%.
4. **Random forest = bagging + per-node random feature subset.** *Two* sources of randomness. Feature subset is re-drawn **at every node**, not once per tree (that's random *subspaces*).
5. **$m$ rule of thumb: $\sqrt N$ (classification), $N/3$ (regression).** But sklearn's `RandomForestRegressor` default is *all* features, not $N/3$.
6. **Hard vs. soft voting:** soft averages probabilities (weights confidence) and usually wins; needs `predict_proba` (so `SVC(probability=True)`).
7. **Decision stump = `max_depth=1`**, a weak learner, AdaBoost's default base estimator.
8. **AdaBoost $\alpha_j = \eta\log\frac{1-r_j}{r_j}$**; can be **negative** if a predictor is worse than random. Misclassified weights multiply by $e^{\alpha_j}$, then renormalize.
9. **AdaBoost prediction is a *weighted* majority** ($\arg\max_k \sum_{j:\hat y_j=k}\alpha_j$), not a plain count.
10. **Learning rate ↓ ⇒ need more estimators.** The boosting step-size trade-off.
11. **Extra-Trees:** random split points → faster, lower variance, higher bias. (Bagging→RF→Extra-Trees = increasing-randomness ladder.)
12. **Stacking blender must be trained on out-of-sample base predictions** or it overfits.
13. **Feature importance** = average impurity reduction per feature, weighted by samples; biased toward high-cardinality features.
14. **Random forest is a black box** — the interpretability cost of averaging.

---

## 🔁 Self-check (predict-first active recall)

Cover the notes and answer aloud before peeking:

1. A colleague bags 500 trees and gets almost no improvement over one tree. Diagnose two likely causes in terms of the slide-7 conditions.
2. Derive the ~37% OOB figure from scratch. What's the in-bag percentage?
3. Why does a random forest usually beat plain bagged trees? Use the word "correlation."
4. You set `learning_rate=0.1` in AdaBoost and accuracy drops. What single other hyperparameter would you change, and which direction?
5. Write the AdaBoost predictor-weight formula and state what $\alpha_j$ does when $r_j = 0.5$ and when $r_j = 0.7$.
6. Soft voting beat hard voting on your data. Give the one-sentence reason.
7. Place these on the bias–variance ladder from most-bias to least-variance: single deep tree, bagged trees, random forest, extra-trees.
8. Why must a stacking blender be trained on held-out predictions? What goes wrong otherwise?
9. One sentence: the fundamental difference in *how* bagging and boosting build their members.
10. Random forests are accurate but you can't explain a single prediction to a regulator. Name the trade-off and the exact mechanism that causes it.

*(Answers are all in the slide-order notes above — grade yourself, then re-read only the ones you missed.)*

---

## 📝 Classwork tracker (attempt independently, then I'll check)

| # | Slide | Task | Status |
|---|---|---|---|
| 1 | 11 | Generate 3 bootstrap samples (`random.choices`) | ⬜ your turn |
| 2 | 16 | Same + identify OOB samples (`set` difference) | ⬜ your turn |
| 3 | 25 | `RandomForestClassifier(100)` on `autompg.csv` → predict Cylinders, drop name col. ~~Visualize a tree~~ (struck by instructor) → **feature importances + per-class report** instead | ✅ done & verified (notebook delivered) |
| 4 | 40 | AdaBoost on Iris (4 features, 80/20), sweep learning rate | ⬜ your turn |

When you've taken a pass at these, send me your code/answers and I'll check them against the instructor's idioms and flag anything off.

---

## 🔎 Cross-check & discrepancy log

**Verified correct (this is the good news — the deck is clean):**
- ✅ **AdaBoost equations (slide 36)** — match Géron's weighted-error, predictor-weight, and instance-update equations. The slide's $r_j$ shows only the numerator, but that's valid because weights are normalized to sum to 1.
- ✅ **OOB ≈ 37% (slide 15)** — confirmed $(1-1/n)^n \to e^{-1} = 0.368$ numerically ($n{=}1000 \to 0.3677$).
- ✅ **Biased-coin ~25% wrong over 1000 tosses (slide 8)** — confirmed ≈ 0.253 via normal approximation with $p=0.51$.
- ✅ **Rule-of-thumb $m$ (slide 23)** — $\sqrt N$ / $N/3$ are Breiman's standard values.

**Nuances to be aware of (NOT slide errors):**
- ⚠️ **`RandomForestRegressor` default ≠ $N/3$.** scikit-learn's regressor defaults to `max_features=1.0` (all features); the $N/3$ on slide 23 is Breiman's theoretical rule. Classifier default *is* `'sqrt'` = $\sqrt N$, matching the slide. (See slide-23 note.)
- ⚠️ **Soft voting needs `predict_proba`.** The slide-6 example uses default hard voting; switching to soft requires `SVC(probability=True)`. (See slide-5 note.)
- ⚠️ **Stacking blender training.** Slide 42 doesn't mention that the blender must see out-of-sample base predictions to avoid overfitting — important for the classwork/notebook. (See slide-42 note.)

**Minor metadata flags:**
- ✅ **Notebook number (slide 26) — RESOLVED in slides' favor.** Slide names `06_ensemble_learning_and_random_forests.ipynb`. My earlier flag claimed it should be 07 — but that was checking `handson-ml3` (Keras/TF repo). Your course uses **`handson-mlp`** (Scikit-Learn & PyTorch edition), where ensemble learning *is* notebook **06**. Slide was correct. Lesson logged: check `handson-mlp`, not `ml3`.
- 📌 **Edition citation (slide 43).** The deck cites the 3rd edition (Keras/TensorFlow, 2022); your course textbook is the Scikit-Learn & PyTorch edition (Oct 2025). The ensemble chapter is scikit-learn-based and essentially identical across editions, so no content impact. *(Note: the Data Preprocessing deck slide 67 cites the correct 2025 PyTorch edition — so the edition citation is inconsistent across decks, not uniformly wrong.)*
- 📌 **Day label.** You called this "Day 6," but our notes have Decision Trees as Day 6 — this looks like the **Day 7** session. Adjust the header to match your repo's numbering if needed.

---

## 🔗 Source-traceability map (built after reading the full Data Preprocessing deck)

The Random Forest classwork (slide 25) pulls preprocessing tools that live in the **Data Preprocessing deck**, not the Ensemble deck. For the AI-free final, here's exactly what is citable to *your* materials versus what is outside knowledge.

**Traceable to YOUR slides (safe to use and cite):**

| Tool / concept | Your source | Notes |
|---|---|---|
| `train_test_split(test_size=0.2)` | Preprocessing **slide 41** | random sampling |
| `stratify=` | Preprocessing **slide 42** | keeps class proportions — directly relevant to the imbalanced cylinders target |
| Drop rows / drop column / impute (the 3 missing-data options) | Preprocessing **slides 50–51** | `dropna`, `drop(axis=1)`, `SimpleImputer(strategy="median")` all shown |
| `OrdinalEncoder` (integer encoding) | Preprocessing **slide 53** | = what your already-numeric `origin` effectively is |
| `OneHotEncoder(handle_unknown="ignore")` | Preprocessing **slide 54** | "one column per value → sparse matrix" |
| `Pipeline` + `ColumnTransformer` | Preprocessing **slides 64–65** | the exact `make_pipeline(SimpleImputer…, OneHotEncoder…)` idiom |
| The peer's whole approach | Preprocessing **slide 66 classwork** | "20% test, median impute, one-hot, ratios, standardize, ideally one pipeline" |
| `MinMaxScaler` / `StandardScaler` | Preprocessing **slides 57–61** | scaling (not needed for trees — see below) |
| Feature importance | Ensemble **slide 27** | impurity reduction averaged across trees |
| "Many features, few relevant → weak trees" | Ensemble **slide 22** | the one-hot-on-high-cardinality risk links here |
| OOB = free validation | Ensemble **slide 15** | OOB accuracy cross-checks the test split |
| Black-box / can't-trace-logic | Ensemble **slide 29** | *why* the tree-viz was struck |
| Pearson "linear only" | Preprocessing **slide 44** | matches your existing Pearson-vs-Spearman note |

**NOT in your slides (general ML / sklearn — verify independently, don't assume examinable):**
- "Trees don't need feature scaling." Your deck teaches scaling for *most* algorithms (slides 56–61) but never states the tree exception. Comes from CART threshold-split mechanics + general knowledge.
- "One-hot *fragments* importance across dummy columns" (the origin 0.023/0.010/0.004 split). Demonstrated on your data; not stated in any slide.
- "One-hot can *hurt* tree ensembles on high-cardinality categoricals." The slide-22 link is a fair bridge, but the claim is general knowledge (Géron ch. 2 / sklearn docs).
- "0.988 vs 0.975 is just noise." Purely empirical, computed on your file.

> 🔴 **NEW slide error found while reading the full Preprocessing deck — slide 36 terminology.** The slide states "Numerical … Also called Ordinal variables" and "Categorical … Also called Nominal variables." This conflates two different axes. Standard taxonomy: **ordinal is a *subtype of categorical*** (ordered categories: Freshman < Junior < Senior), distinct from **numerical/continuous**. The slide's own **slide-55 classwork** treats `Standing` (Freshman/Junior/Senior) as categorical-for-encoding — and `OrdinalEncoder` (slide 53) exists precisely to encode *ordered categoricals*, which makes no sense under the slide-36 labeling. Flag inline: numerical ≠ ordinal; ordinal ⊂ categorical. **(Confirm against the textbook's variable-types section.)**

---

## 🛠️ Consolidated `skills.md` patch (for session close)

`skills.md` wasn't re-uploaded this session, so this isn't applied yet — upload it and I'll merge. Grouped by section for a clean patch.

**Ensemble learning (Day 7 content):**
- **Bias–variance map:** Bagging/RF/Extra-Trees → reduce **variance** (parallel, independent, high-variance members). Boosting → reduces **bias** (sequential, dependent, weak members). The "which family" answer key.
- **OOB derivation is fair game:** $(1-1/n)^n \to e^{-1} \approx 0.368$; in-bag ≈ 63%. Derive, don't just recite. OOB accuracy cross-checks the held-out test number.
- **Random forest = TWO randomness sources:** bootstrap rows **+** per-node random feature subset (re-drawn every node, *not* once per tree → that's random subspaces).
- **sklearn default trap:** `RandomForestRegressor` defaults to *all* features, **not** $N/3$. Classifier default is `'sqrt'` = $\sqrt N$ (matches slide 23). Distinguish library default from Breiman's rule.
- **AdaBoost math:** $\alpha_j=\eta\log\frac{1-r_j}{r_j}$ (negative if worse-than-random); misclassified weights ×$e^{\alpha_j}$ then renormalize; prediction is **weighted** majority.
- **Boosting learning-rate trade-off:** smaller `learning_rate` ⇒ need more `n_estimators`.
- **Stacking blender hygiene:** train on out-of-fold / held-out base predictions or it overfits.

**Preprocessing cross-references (so future classwork doesn't re-derive these):**
- **Preprocessing tool → slide index:** split 41 · stratify 42 · missing-data 3 options 50–51 · OrdinalEncoder 53 · OneHotEncoder 54 · Pipeline+ColumnTransformer 64–65 · scaling 57–61 · full-pipeline classwork 66.
- **For tree models specifically:** scaling is unnecessary; one-hot is *optional* and can fragment importances / hurt on high-cardinality (slide-22 risk). Integer/ordinal `origin` is fine as-is for a forest.

**Canvas dataset schema (pinned):**
- **`autompg.csv`:** comma-delimited, quoted fields; columns `mpg, cylinders, displacement, horsepower, weight, acceleration, model_year, origin, car_name`; `origin` **numeric (1/2/3)**; `horsepower` has **6 real `NaN`s** (not `'?'` strings); last column `car_name`. After drop+dropna: **392×7**, classes {3,4,5,6,8}, heavily imbalanced.

**Reference-repo & idiom notes:**
- **Textbook repo is `handson-mlp`** (Scikit-Learn & PyTorch, Oct 2025), *not* `handson-ml3` (Keras/TF). Ensemble notebook = **06** there. Always cross-check notebook numbers against `handson-mlp`.
- **Géron forest idioms:** `rnd_clf` naming, `n_jobs=-1`, `random_state=42`; feature-importance print loop `for score, name in zip(rnd_clf.feature_importances_, cols): print(round(score,2), name)`.

**Slide error log (append):**
- Ensemble deck: **clean** (AdaBoost eqns, OOB 37%, biased-coin 25%, $m$ rules all verified). Notebook-06 flag resolved in slides' favor.
- Preprocessing deck **slide 36**: numerical ≠ ordinal; ordinal is a subtype of categorical. Contradicts slides 53/55. *(confirm vs textbook)*

**Standing principle reaffirmed:** document *clean* cross-check results too, not only errors, so the running log stays trustworthy. And: separate "in your slides" from "general knowledge I added" explicitly — matters for the AI-free final.
