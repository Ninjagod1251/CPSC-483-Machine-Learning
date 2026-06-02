# CPSC 483 — Day 3 Notes
## Classification & Performance Measures (Géron, Ch. 3)

---

## 1. Datasets 101

Researched five classic datasets — creator, size, and target.

| Dataset | Created by | Rows | Columns | Target / purpose |
|---|---|---|---|---|
| **Iris** | Measurements by botanist **Edgar Anderson** (1935); analyzed & popularized by **R. A. Fisher** (1936) | 150 (50 per species) | 5 (4 features + species) | Predict **species**: setosa / versicolor / virginica (3-class) |
| **MNIST** | **LeCun, Cortes, Burges** — built from NIST digit databases | 70,000 (60k train + 10k test) | 784 features (28×28 pixels) + 1 label | Predict **digit 0–9** (10-class image classification) |
| **Pima Indians Diabetes** | **NIDDK** (National Institute of Diabetes and Digestive and Kidney Diseases); donated to UCI by V. Sigillito | 768 | 9 (8 features + outcome) | Predict **Outcome** = diabetes within 5 yrs (binary) |
| **Auto MPG** | **StatLib (Carnegie Mellon)**; UCI version attributed to **Ross Quinlan (1993)** | 398 (or 392 after dropping missing horsepower) | 9 (mpg + 7 features + car name) | Predict **mpg** (regression) |
| **Titanic** | Passenger records (Encyclopedia Titanica / Vanderbilt "titanic3"); popularized by **Kaggle** (2012) | 891 (Kaggle `train.csv`); 1,309 full | 12 (`train.csv`) | Predict **Survived** (binary) |

**Gotchas worth a sentence in any writeup:**
- *Auto MPG* has 398 raw rows but 6 have missing `horsepower`, so many notebooks report 392. State which you used.
- *Titanic* `train.csv` = 891 rows; `test.csv` has only 11 columns because the **Survived** target is withheld. Full passenger list = 1,309.
- "Number of columns" is ambiguous — always say whether it includes the target.
- *MNIST* "rows" really means images; each is a flattened 784-value vector (`X.shape == (70000, 784)`).

---

## 2. MNIST — plotting a specific digit

The textbook plots `X[0]`, which is a **5**. To plot a **7**, reuse the already-loaded data:

```python
seven_index = (y == '7').argmax()   # first index where the label is "7"
plot_digit(X[seven_index])
plt.title(f"Label: {y[seven_index]}")
plt.show()
```

**Key gotchas:**
- `fetch_openml('mnist_784')` returns labels as **strings** → compare to `'7'`, not `7`. Comparing to the int `7` matches nothing; `argmax` then silently returns index 0 (the 5).
- `.argmax()` on a boolean array returns the **first True** index — handy for "first instance of class X." (`np.where(y=='7')[0][0]` does the same.)
- `image_data.reshape(28, 28)` works because the 784 features are the image flattened row-by-row.

---

## 3. Performance measures — the core lesson

### Confusion matrix (positive = the class of interest)

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

### Metrics

- **Accuracy** = (TP + TN) / N — fraction correct
- **Error rate** = 1 − Accuracy = (FP + FN) / N
- **Precision** = TP / (TP + FP) — of what we *flagged*, how much was right
- **Recall** (sensitivity) = TP / (TP + FN) — of what was *truly positive*, how much we caught
- **F1** = 2 · (P · R) / (P + R) — harmonic mean; punishes a low value in either

### Precision/Recall trade-off
A classifier scores each instance; you compare the score to a **threshold**.
- **Lower** threshold → predict positive more often → **recall ↑, precision ↓**
- **Raise** threshold → more selective → **precision ↑, recall ↓**
You cannot maximize both at once. Choose based on which error is worse.

### Why accuracy alone misleads (imbalanced classes)
If positives are rare (say 1 in 10), a model that always predicts "negative" is ~90% accurate and catches **zero** positives (recall = 0).
- Textbook example: the **"never-5" dummy classifier** is ~90% accurate on MNIST 5-detection yet useless.
- → This is *why* Ch. 3 drops accuracy and moves to confusion matrix / precision / recall.

---

## 4. Worked classwork — Credit / Income / Risk

Data (positive class = **High Risk**):

| Credit | Income | Actual |
|---|---|---|
| 600 | 50,000 | High |
| 650 | 60,000 | Low |
| 800 | 55,000 | Low |
| 550 | 55,000 | Low |
| 660 | 50,000 | High |
| 750 | 58,000 | Low |

Actual positives (High) = 2 (rows 1, 5); actual negatives (Low) = 4; N = 6.

### Classifier 1 — "Risk is always High"
Predicts High for everyone → TP = 2, FP = 4, FN = 0, TN = 0.

| Metric | Value |
|---|---|
| Accuracy | 2/6 = **33.3%** |
| Error rate | **66.7%** |
| Precision | 2/6 = **33.3%** |
| Recall | 2/2 = **100%** |
| F1 | **0.50** |

### Classifier 2 — Low if (Credit > 700 OR Income ≥ 60,000), else High
Predictions: High, Low, Low, High, High, Low.
- Row 1 High→High = TP · Row 2 Low→Low = TN · Row 3 Low→Low = TN
- Row 4 High(pred)→Low(actual) = FP · Row 5 High→High = TP · Row 6 Low→Low = TN

TP = 2, TN = 3, FP = 1, FN = 0.

| Metric | Value |
|---|---|
| Accuracy | 5/6 = **83.3%** |
| Error rate | **16.7%** |
| Precision | 2/3 = **66.7%** |
| Recall | 2/2 = **100%** |
| F1 | **0.80** |

### The insight
Both classifiers hit **100% recall**, so recall alone calls them equal. But Classifier 1 gets that recall trivially by predicting High for everyone (FN = 0 because it never predicts Low). **F1 separates them (0.50 vs 0.80)** because it punishes Classifier 1's terrible precision. Same trap as the "never-5" dummy: one metric on imbalanced data hides a degenerate model.

---

## 5. Real-world thread — thresholds, FP/FN asymmetry, and recalls

This tied the whole confusion-matrix lesson to real safety regulation.

### The asymmetric cost of FP vs FN (defect / recall framing)
Positive = "has a real safety defect."
- **FN (missed defect)** = catastrophic — defective product stays in use; people can be hurt.
- **FP (false alarm)** = expensive/embarrassing — unnecessary recall — but no one is harmed.
- Because FN ≫ FP in cost, safety systems tune for **high recall**, accepting lower precision.

### Cars — no fixed numeric threshold
- A recall is a **regulatory/legal decision**, not a model output. NHTSA's standard is qualitative: a *safety-related defect* posing an *unreasonable risk*.
- ML / statistical analysis is used in the **screening** stage to flag defect trends from complaints, warranty claims, and Early Warning Reporting data (NHTSA explicitly hires for ML for this). Humans then decide.
- Pattern = **two-stage cascade**: screening (high recall, catch everything suspicious) → formal investigation (high precision, rule out false alarms before forcing a recall).

### Food — yes, explicit numeric thresholds
- FDA **Defect Action Levels (DALs)** are literal cutoffs for unavoidable contamination; at/above them, food is deemed "adulterated."
  - e.g., peanut butter: avg < 30 insect fragments / 100 g; canned mushrooms: > 20 maggots + 75 mites / 100 g.
- These are set **loosely** because the defects are *aesthetic* (low harm) → optimize for **precision** (don't destroy edible food).
- **Pathogens flip it**: Listeria in ready-to-eat food, E. coli O157:H7 in ground beef → **zero tolerance** (threshold at the detection limit) → maximize **recall** because a miss can kill.

### Unifying idea (memorize this)
> A regulatory limit is a **decision threshold**, and regulators set it according to how bad a **false negative** is. Cosmetic defect → loose threshold (favor precision). Lethal pathogen → zero tolerance (favor recall). Cars → no scalar threshold, so expert judgment replaces the cutoff.

---

## 6. Debugging lesson — three categories of "it's broken"

When code "doesn't work," classify the problem:
1. **Bug** — the code is wrong.
2. **Environment issue** — code is correct but a library/version breaks it (e.g., `fetch_openml` timing out, a deprecated matplotlib argument).
3. **Task mismatch** — code is correct but doesn't do *your* assignment (e.g., it plots `X[0]` = a 5 when you needed a 7).

Today's notebook was case **3**: it was byte-for-byte the official textbook notebook (no bug). The fix was to *add* a cell that selects a 7 — not to switch data sources.

**Watch-out:** loading MNIST from `tensorflow.keras` instead of `fetch_openml` works, but (a) it's a heavy extra dependency for one plot, and (b) it **overwrites** `X, y, X_train, ...`, which breaks later cells. Prefer reusing the data already loaded.

---

## 7. Self-check (answer unaided)

1. In a safety-recall setting, which error do you minimize, and which metric does that map to?
2. A model is 99% accurate at detecting a defect present in 1% of units. Why might it still be useless?
3. Why must MNIST labels be compared with `'7'` and not `7`?
4. Both classifiers in §4 had 100% recall. Which metric exposed the bad one, and why?
5. Why is the Listeria threshold zero but the insect-fragment threshold high?

---

## 8. Carry-forward
- Next: **Linear & Logistic Regression (Ch. 4)** — precision/recall, ROC, thresholds will recur.
- Practice predict-before-run on confusion-matrix and threshold cells.
- Keep the FP/FN asymmetry + "threshold = decision tuned by FN cost" framing handy for the midterm.
