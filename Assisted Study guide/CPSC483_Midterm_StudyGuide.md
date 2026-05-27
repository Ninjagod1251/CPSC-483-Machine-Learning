# CPSC 483 — Midterm Open-Note Study Guide
### Covers AG Chapters 1–7 + SVM (Appendix C) · Exam: Wed 6/11

> Confirm with Dr. Panangadan whether notes are actually permitted — the syllabus says exam aids are "as described by the instructor." If allowed, this is your sheet. Either way it's your review map.

---

## 1 — The Machine Learning Landscape (Ch. 1)

**ML =** a program whose performance on a task improves with experience (data) rather than explicit rules.

**Taxonomy by supervision**
| Type | Training signal | Typical tasks |
|---|---|---|
| Supervised | labeled data | classification, regression |
| Unsupervised | no labels | clustering, association, anomaly detection, dim. reduction |
| Semi-supervised | few labels + many unlabeled | photo tagging |
| Self-supervised | labels generated from the data itself | masked-image/word prediction, LLM pretraining |
| Reinforcement | reward signal from environment | game agents, control |

**Other axes:** batch vs **online** (incremental) · **instance-based** (compare to stored examples, e.g. KNN) vs **model-based** (fit parameters, then predict).

**Core problems**
- **Overfitting:** model too complex / too little data → great on train, poor on test. Fixes: simpler model, more data, regularization.
- **Underfitting:** model too simple → poor everywhere. Fixes: more powerful model, better features, less regularization.
- **Generalization** is the goal; estimate it with a held-out **test set**.
- **No Free Lunch theorem:** with zero assumptions about the data, no model is inherently best — you must try a few reasonable ones.

---

## 2 — End-to-End Project & Data Preparation (Ch. 2)

**Pipeline:** frame problem → get data → **set aside test set** → explore (training only) → prepare → model → fine-tune → present → deploy.

- **Make the test set first** (~20%) and never look at it during exploration — looking causes **data snooping bias**.
- **Stable splits:** seed the RNG, or hash each instance's stable ID so the split survives dataset updates.
- **Stratified sampling:** split so important categories keep the same proportions as the full dataset (vs purely random, which can skew small datasets).

**Cleaning & transforming**
- **Missing values:** drop rows, drop column, or **impute** (e.g. median). `SimpleImputer`.
- **Categorical → numeric:** ordinal encoding (ordered) or **one-hot** (unordered, most common).
- **Feature scaling** (most algorithms need it):
  - **Min-max / normalization** → range [0,1].
  - **Standardization** → mean 0, std 1; less sensitive to outliers.
- **Heavy-tailed features:** log-transform toward Gaussian-ish.
- **Pipelines** (`Pipeline`, `ColumnTransformer`) chain transforms so the same prep applies to train, validation, and new data — prevents leakage.

**Validation:** **k-fold cross-validation** = split train into k folds, train on k−1, validate on the held-out fold, rotate. Gives a mean ± std performance estimate without touching the test set.

---

## 3 — Classification & Performance Measures (Ch. 3)

**Confusion matrix** (binary): TP, FP, TN, FN.

| Metric | Formula | Reads as |
|---|---|---|
| Accuracy | (TP+TN)/all | misleading on imbalanced data |
| **Precision** | TP/(TP+FP) | of predicted-positive, how many correct |
| **Recall** (sensitivity, TPR) | TP/(TP+FN) | of actual-positive, how many caught |
| **F1** | 2·(P·R)/(P+R) | harmonic mean; punishes imbalance between P and R |

- **Precision/recall trade-off:** moving the decision threshold raises one and lowers the other.
- **PR curve:** precision vs recall; prefer when positives are rare.
- **ROC curve:** TPR vs FPR; **AUC** = area under it (1.0 perfect, 0.5 random).
- **Multiclass:** one-vs-rest (OvR) or one-vs-one (OvO). **Multilabel** = multiple binary tags per instance. **Multioutput** = multiple label values.

---

## 4 — Training Models (Ch. 4)

**Linear regression** ŷ = θ₀ + θ₁x₁ + … = **θᵀx**. Cost = **MSE**.
- **Normal Equation / SVD:** closed-form solution; great for few features, slow when features are many.
- **Gradient Descent:** iteratively step downhill on the cost gradient. **Learning rate** too high → diverge; too low → slow.
  - **Batch GD:** whole set each step (stable, slow on big data).
  - **Stochastic GD (SGD):** one instance per step (fast, noisy; can escape local minima).
  - **Mini-batch GD:** small batches (practical middle ground).
  - **Always scale features** before GD or it zig-zags.

**Polynomial regression:** add powers of features, then fit linearly → can model curves (and overfit).

**Bias–variance trade-off:** total error ≈ bias² + variance + irreducible noise. Underfit = high bias; overfit = high variance.

**Regularization** (shrinks weights to fight overfitting):
- **Ridge (L2):** penalty Σθ² — shrinks weights smoothly.
- **Lasso (L1):** penalty Σ|θ| — drives some weights to **exactly 0** (feature selection).
- **Elastic Net:** mix of L1 + L2.
- **Early stopping:** halt training when validation error starts rising.

**Logistic regression:** outputs probability via **sigmoid** σ(t)=1/(1+e⁻ᵗ); classify by threshold (0.5). Cost = **log loss / cross-entropy**.
**Softmax regression:** multiclass generalization — one score per class, normalized to probabilities.

---

## SVM — Support Vector Machines (Appendix C)

- Goal: find the decision boundary with the **largest margin** between classes ("widest street"). The boundary is set by the **support vectors** (the instances on the margin edge).
- **Hard margin:** no violations allowed — fails with outliers / non-separable data.
- **Soft margin:** allow some violations; hyperparameter **C** controls the trade-off. **Small C → wider margin, more violations** (more regularization); large C → narrower, fewer violations.
- **Kernel trick:** implicitly map to higher dimensions to separate non-linear data without computing the mapping. Common: **polynomial**, **RBF (Gaussian)**. RBF's **γ** controls reach — high γ = wiggly/overfit, low γ = smooth.
- **Always scale features.** SVMs do regression too (SVR).

---

## 5 — Decision Trees (Ch. 5)

- Split data by feature thresholds to make nodes as **pure** as possible. CART (binary splits) in scikit-learn.
- **Impurity measures:** **Gini** (default, fast) or **entropy** (information gain). Both ~similar.
- **White-box / interpretable;** needs little data prep (no scaling required).
- **Prone to overfitting** → regularize with `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`.
- **High variance:** small data changes → very different tree. (This motivates ensembles.)
- Do regression too (predict the average target in each leaf → step-function output).

---

## 6 — Ensemble Learning (Ch. 6)

Combine many weak/diverse models → stronger one ("wisdom of the crowd").

- **Voting:** aggregate different models. **Hard** = majority vote; **soft** = average predicted probabilities (usually better).
- **Bagging / Pasting:** same algorithm on random subsets of *instances* (bagging = with replacement). **Out-of-bag (OOB)** samples give a free validation estimate.
- **Random Forest:** bagged trees + randomness in *feature* selection at each split → decorrelated trees, lower variance. Gives **feature importance**.
- **Boosting** (sequential, each model fixes predecessor's errors):
  - **AdaBoost:** re-weights misclassified instances each round.
  - **Gradient Boosting:** each new tree fits the **residual errors** of the ensemble so far. (XGBoost = optimized version.)
- **Stacking:** train a "blender" model to combine base predictions.
- Rule of thumb: bagging/RF reduce **variance**; boosting reduces **bias** (but can overfit if over-trained).

---

## 7 — Dimensionality Reduction (Ch. 7)

**Why:** curse of dimensionality (data sparse in high-D), speed, visualization, noise reduction.

- **Projection** vs **Manifold learning** (data lies on a lower-D curved surface).
- **PCA:** find orthogonal axes (**principal components**) capturing maximum variance; project onto top components. Uses **SVD**. Choose #components by **explained variance ratio** (e.g. keep 95%). **Must scale/center first.**
- **Variants:** Randomized PCA (faster), **Incremental PCA** (streaming/large data), **Kernel PCA** (non-linear).
- **Other:** **LLE** (manifold, non-linear), **t-SNE** (visualization, keeps local clusters — not for downstream modeling).
- Trade-off: reduces compute & noise but loses some information.

---

## Fast self-test (answer unaided — exams are AI-free)
1. Precision vs recall — define each and when you'd optimize for which.
2. Why scale features before SVM/GD/PCA but not for decision trees?
3. Bagging vs boosting — what error does each primarily reduce?
4. Lasso vs Ridge — which zeros out weights and why?
5. What is data snooping bias and how do you avoid it?
6. Why does soft-margin SVM use C, and which direction widens the margin?
7. How do you pick the number of PCA components?
8. Stratified vs random sampling — when does it matter?
9. Sigmoid vs softmax — when is each used?
10. What does AUC of 0.5 mean?

---
*Synthesized 2026-05-26 from standard ML concepts aligned to Géron's framing. Verify exam scope and the notes policy with the instructor.*
