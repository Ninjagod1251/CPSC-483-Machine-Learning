# Day 2 Notes — Data Preprocessing
### CPSC 483 · Tue 5/27 · Dr. Panangadan · 67 slides · AG Ch.2

> Lecture covers: framing an ML project, getting and exploring data with pandas, splitting into train/test the right way, transforming and scaling features, handling missing values and categorical variables, and wrapping the whole thing in a scikit-learn pipeline.

---

## 1. The typical ML project — eight steps

Before getting into preprocessing techniques, the lecture frames where they fit. A real ML project follows roughly this sequence:

1. **Look at the big picture** — understand the problem and its context.
2. **Get the data.**
3. **Explore and visualize the data** to gain insights.
4. **Prepare the data** for ML algorithms. *(← bulk of today's lecture)*
5. **Select a model and train it.**
6. **Fine-tune the model.**
7. **Present the solution.**
8. **Launch, monitor, maintain.**

Today's preprocessing work is step 4, but with a critical chunk of step 3 (EDA) thrown in — because how you prepare the data depends on what you find while exploring it.

## 2. Look at the big picture

The lecture's running example is the **California housing dataset**: use census data (population, median income, etc., per block group) to predict median house value per district.

Two questions to anchor any new ML project:

**A. What is the overall objective?**
Building the model is *not* the end goal — it has to feed some downstream decision. The objective determines:
- Which performance measure to use.
- How accurate the model has to be.
- What an acceptable error looks like.

**B. What does the current solution look like (if any)?**
Whatever the company / lab / process currently does gives you a baseline. If their current approach is "expert intuition giving ±30% error," your bar is set.

## 3. Recap: types of ML (carry-over from Day 1)

The deck re-presents the taxonomy because preprocessing choices depend on it:

- **By training supervision** — supervised / unsupervised / semi-supervised / self-supervised / reinforcement.
- **By output type** — classification / regression / clustering.
- **By data volume and rate** — batch (all data at once) vs. online (streaming).

The housing problem is **supervised regression, batch learning.**

## 4. Performance measures (for regression)

Two standards introduced — both relevant before the final model exists, because the metric drives validation.

- **Root Mean Squared Error (RMSE):** square the errors, average, take the square root. Penalizes large errors heavily (because of the squaring). Same units as the target → easy to interpret ("off by ~$50,000 on house price").
- **Mean Absolute Error (MAE):** average of the absolute errors. More forgiving of outliers than RMSE.

When the dataset has many outliers, MAE may be the more honest metric. When you want to penalize big mistakes more than small ones, RMSE.

---

## 5. Where to get data

Common public repositories named in the slides:
- **OpenML.org** — community ML datasets, also accessible from sklearn via `fetch_openml`.
- **Kaggle Datasets** — competitions + thousands of datasets.
- **UCI Machine Learning Repository** — classic, well-curated.
- **AWS Open Data**, **TensorFlow Datasets**.
- Meta-portals: **DataPortals.org**, **Data.gov**.
- Also: Wikipedia's ML dataset list, Quora, the `r/datasets` subreddit.

Useful to know for the course project (real-world dataset required).

## 6. The Python ecosystem (quick orientation)

Why Python: high-level, object-oriented (everything is an object), open-source, "batteries included," supports JIT and compilation.

### Key packages
| Package | What it does |
|---|---|
| **NumPy** | Fast numerical arrays; 10×+ faster than pure Python lists |
| **pandas** | Loading, structuring, cleaning, indexing tabular data |
| **SciPy** | Scientific computation (optimization, stats, signal processing) |
| **Matplotlib** | Plotting |
| **scikit-learn** | Classical ML — built on numpy/pandas/scipy/matplotlib |
| **PyTorch** | Neural nets / deep learning (covered later in this course) |
| **Keras** | High-level deep-learning API |
| **statistics** | Stdlib statistics functions |

### Environments
- **Jupyter Notebooks** — interleave code, output, prose, and math (Markdown + LaTeX). Standard for data science.
- **Google Colab** — Google's hosted Jupyter, with free GPU/TPU access. Used throughout this course; no setup, runs in browser, easy to share.

Other ML stacks acknowledged but not used here: MATLAB, R, SAS/SPSS, ML.NET, Weka, AWS SageMaker, Azure ML, Google ML.

---

## 7. pandas — the working tool for tabular data

### Two core structures
- **Series** — one column of data (1-D).
- **DataFrame** — a 2-D table; collection of Series with different dtypes per column. Mutable in size, labeled axes.

### Creating a DataFrame
From a dictionary, list, ndarray, Series, or another DataFrame:
```python
import pandas as pd
data = {'apples':[3, 2, 0, 1], 'oranges':[0, 3, 7, 2]}
df = pd.DataFrame(data)
```

From a CSV file:
```python
df = pd.read_csv("housing.csv")
```

Sklearn datasets like iris come ready-to-use:
```python
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
my_df = iris.data
```

### Inspecting a DataFrame
| Call | Purpose |
|---|---|
| `df.info()` | dtypes + non-null counts per column |
| `df.describe()` | mean, std, min, quartiles, max for numeric cols |
| `df.head(n)` / `df.tail(n)` | first/last n rows |
| `df.shape` | (nrows, ncols) |
| `len(df)` | nrows |

### Subsetting — label-based with `.loc`
```python
df.loc[[3, 4], ["population", "households"]]     # specific rows + cols by label
df.loc[df.total_rooms > 7000, ["population", "households"]]  # filter rows
df.loc[3:4, ["population", "households"]]        # label slice — end inclusive!
```

### Subsetting — position-based with `.iloc`
```python
df.iloc[3:4, 5:6]    # integer rows + cols
df.iloc[3:5, 5:7]    # integer slice — end exclusive (Python-standard)
```

The end-inclusive vs end-exclusive difference is the trap.

### Quick shortcuts
- Single column: `df.longitude` or `df["longitude"]` → Series.
- Multiple columns: `df[["longitude", "latitude"]]` → DataFrame.
- Row slice: `df[0:3]`.
- Filter rows: `df[df.total_rooms > 7000]`.

> **Class exercise (slide 33):** practice on `housing_full` — get first 5 columns; just `total_rooms` and `total_bedrooms`; rows with population > 1000; that filter combined; alternate rows.

---

## 8. Exploratory Data Analysis (EDA)

### Purpose
- Get a general sense of the data before modeling.
- Data-driven and model-free — you're not predicting anything yet.
- Interactive and visual — humans are good at pattern recognition.

### What EDA systematically checks
- **Summary statistics** (mean, median, deviations).
- **Data quality** — missing values, outliers, impossible values.
- **Each variable's distribution** (single-variable viz).
- **Pairwise relationships** between variables.

### Summary stats
```python
df.describe()
```
Returns count, mean, std, min, quartiles, max for each numeric column.

### Data quality checks
- **Missing data:** `df.isna()` to flag, `df.isna().sum()` to count per column.
- **Sneaky missing values:** missing values aren't always stored as NaN. Some datasets use `0`, `-1`, or `999` as a sentinel for "no answer." Sanity-check against domain knowledge (can a person really weigh 0 kg?).
- **Outliers:** values far from the rest of the distribution. Check per-variable using histograms or box plots.

### Numerical vs. categorical variables
| Type | Ordering? | Examples | Statistics that work |
|---|---|---|---|
| **Numerical / ordinal** | natural order | age, weight | mean, median, std |
| **Categorical / nominal** | no order | marital status, color | counts, mode |

This distinction governs which encoders, scalers, and plots apply.

### Single-variable visualization
| Variable type | Plot |
|---|---|
| Numerical | **histogram** (`df.hist()`), box plot |
| Categorical | **bar chart** |

Histograms reveal center, variability, skewness, modality, outliers, and weird patterns. Two gotchas: **bin width matters** (too few bins hides structure; too many is noisy), and **be careful with real zeros** that might actually be sentinel missing values.

---

## 9. Create the test set — *first thing after loading*

### Why early?
- **Data snooping bias:** if you look at the whole dataset before splitting, you (or your code) start tuning to patterns in what's supposed to be held-out data. The test score then overstates real-world performance.
- Do all EDA *only* on the training subset.

### Typical proportions
- 80% training, 20% test is a standard starting point.
- Training set should be as large as possible (more data → better models).
- Test set should be large enough to be **representative** of the full dataset (features in the same proportions).

### Random sampling
Simple but with one quiet flaw — it isn't stable across dataset updates.
```python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(myfulldata, test_size=0.2)
```
If you re-run after appending new rows, the split changes. You'll start contaminating test data with what used to be training data.

**Fix — split on a unique identifier:**
- Use a row's stable ID, or build one if none exists.
- Example for the housing data: `id = longitude * 1000 + latitude` (lat/lon are stable for millions of years).
- Hash the ID; rows whose hash falls below a threshold go to test. Same row always gets the same hash → split is stable.

### Stratified sampling
Random sampling can give you a test set whose feature proportions differ noticeably from the full dataset — especially when an important feature is unevenly distributed.

**Stratified sampling** guarantees the test set keeps the same proportions of a chosen column as the full dataset:
```python
train_set, test_set = train_test_split(
    myfulldata, test_size=0.2,
    stratify=myfulldata["col_name"]
)
```
For the housing example, median income drives house value, so the test set is stratified on income categories (continuous income → binned via `pd.cut`).

---

## 10. Relationships between variables

After EDA on the training set, look for relationships — which features look predictive of the target?

### Visually — scatter plots
For two numerical variables:
```python
mydata.plot(kind="scatter",
            x="median_income", y="median_house_value",
            alpha=0.1, grid=True)
```
- `alpha=0.1` makes overlapping points visible as density (essential for large datasets).
- Encode a third or fourth variable via point size and color.

### Numerically — correlation
```python
corr_matrix = mydata.corr(numeric_only=True)
corr_matrix["output_var"].sort_values(ascending=False)
```
- Pearson correlation in [-1, +1].
- **High negative is also strong** — a feature that goes *down* as the target goes up is just as useful as one that goes up.
- Near 0 means no *linear* relationship — there could still be a non-linear one the correlation coefficient misses.

---

## 11. Feature transforms

The data you have isn't always the data you want. Two reasons to transform features:

### A. Apply a numerical function
```python
housing["log_total_rooms"] = np.log(housing["total_rooms"])
```
Common transforms:
- **log()** for long-tailed (right-skewed) features — pulls them toward a Gaussian-ish shape, which most ML models prefer.
- **sqrt(), reciprocal**, etc., depending on shape.

### B. Combine features into new ones
Often a *ratio* of two existing features correlates better with the target than either feature alone:
```python
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
```
For the housing data, `bedrooms_ratio` correlates with house value more strongly than either total_bedrooms or total_rooms individually.

> **Class exercise (slide 47):** create your own derived features (ratios) and see whose correlation with the target you can maximize.

### C. The `FunctionTransformer` wrapper
To use a custom function inside a sklearn pipeline, wrap it:
```python
from sklearn.preprocessing import FunctionTransformer

# Element-wise: log of every value
log_transformer = FunctionTransformer(np.log,
                                       feature_names_out="log transform")

# Custom: ratio of two columns
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

ratio_transformer = FunctionTransformer(column_ratio,
                                         feature_names_out="ratio")
```
This is what lets your hand-crafted transforms drop into a pipeline alongside scikit-learn's built-ins.

---

## 12. Handling missing data

Most ML algorithms cannot handle NaN. You have three options:

| Option | Pandas code | When to use |
|---|---|---|
| **Drop rows** with any missing value | `df.dropna(subset=["bad_feature"], inplace=True)` | When missingness is rare and you can afford to lose rows |
| **Drop the column** that's missing | `df.drop("bad_feature", axis=1, inplace=True)` | When a feature is mostly missing or unimportant |
| **Impute** — replace with an estimate | see below | Most common; preserves all rows |

### Imputation
With pandas directly:
```python
median = df["bad_feature"].median()
df["bad_feature"].fillna(median, inplace=True)
```
With sklearn's `SimpleImputer` (preferred in a pipeline because it remembers the training-set median to reuse on test/new data):
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
```
Strategies: `"median"`, `"mean"`, `"most_frequent"`, `"constant"`. **Median is the default to memorize** — robust to outliers, while mean isn't.

For categorical columns, use `strategy="most_frequent"`.

> **Class exercise (slide 52):** practice all three on a tiny DataFrame with NaNs in three columns.

---

## 13. Categorical → numerical

ML algorithms operate on numbers. Categorical features have to be encoded.

### Option 1: Ordinal encoding — integers per category
```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
Color_encoded = ordinal_encoder.fit_transform(Color)
# Red→0, Blue→1, Green→2, ...
```
**Use only when categories have a real order** (e.g., bad / average / good / excellent). Using ordinal encoding on unordered categories tells the algorithm a false ordering (it thinks Green > Blue > Red), which corrupts learning.

### Option 2: One-hot encoding — one column per category
```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
Color_onehot = cat_encoder.fit_transform(Color)
# Color_Red   = [1,0,0,0,...]
# Color_Blue  = [0,1,0,1,...]
# Color_Green = [0,0,1,0,...]
```
- Default for unordered categories.
- Each row has exactly one "1" (hot); the rest are 0 (cold).
- Output is a sparse matrix (efficient when most entries are zero).

**Pitfall:** with many categories (country codes, professions), one-hot creates many columns, slowing training. Consider replacing the category with a meaningful numeric feature (a country's GDP per capita) or — for neural nets — using embeddings.

**Why `OneHotEncoder` over `pd.get_dummies`?** The sklearn encoder *remembers* which categories it saw at training time, so test/production data always produces the same columns in the same order. `get_dummies` regenerates columns from whatever it sees. Set `handle_unknown="ignore"` to make the encoder tolerant of new unseen categories instead of erroring.

> **Class exercise (slide 55):** encode a small student-roster DataFrame (CWID, Standing, GPA) using both ordinal and one-hot — by hand on paper, then with code. Be able to write out the column layout by hand.

---

## 14. Feature scaling

Most ML algorithms misbehave when features span wildly different ranges — they bias toward features with larger magnitudes regardless of importance. Two scalers:

### Min-Max scaling (normalization)
Linearly rescales each feature independently to [0, 1] (or any range).

Formula per value: (x − min) / (max − min).

Example: in the range [16, 100], the value 37 maps to (37−16)/(100−16) = 0.25.

```python
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
feature_scaled = min_max_scaler.fit_transform(original_feature)
```

**Catch:** sensitive to outliers. A single extreme value stretches the range and compresses every other value into a tiny strip.

> **Class exercise (slide 59):** apply min-max scaling to a small dataset by hand and in code.

### Standardization (Z-score)
Subtract the mean, divide by the standard deviation. Result: mean 0, standard deviation 1.

```python
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
features_scaled = std_scaler.fit_transform(original_features)
```

**Properties:**
- Less sensitive to outliers than min-max.
- Output is **not bounded** — values can be very large positive or negative.
- Preferred for most ML algorithms that assume roughly Gaussian features (linear/logistic regression, SVMs, neural nets).

**Both scalers share a critical rule:** `fit` only on the training set, then `transform` on both train and test. Calling `fit_transform` on the test set leaks test statistics and silently inflates your scores.

---

## 15. Data pipelines — the concept

A **pipeline** is a sequence of data-processing components that each read data, process it, and write output to a store. Properties:

- Components run **asynchronously** — each reads from / writes to a data store.
- Components are **self-contained** — they communicate only through the data store.
- Independent development and maintenance.
- Robust — if one component breaks, downstream can keep running for a while on the last good output.

**Trade-off:** the same isolation that makes pipelines robust also means a broken component can go *unnoticed*. Stale data → quietly degraded outputs. Monitoring matters.

(This is the production-software view. For an in-process scikit-learn pipeline, "components" are sklearn transformers chained in sequence within one Python program.)

### scikit-learn pipelines
Glue all your preprocessing steps together so the same transformations apply identically to train, validation, and new data — preventing leakage and saving a huge amount of boilerplate.

**`make_pipeline`** chains a sequence of transforms:
```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)
```

**`ColumnTransformer`** routes different sub-pipelines to different columns and concatenates the results:
```python
from sklearn.compose import ColumnTransformer, make_column_selector

preprocessing = ColumnTransformer([
    ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
])

housing_prepared = preprocessing.fit_transform(housing)
```

`make_column_selector` lets you pick columns by dtype rather than naming each one — cleaner code and resilient to schema changes.

---

## 16. The big in-class exercise (slide 66) — putting it all together

Build *one* pipeline on the housing data that:
1. Holds out 20% as a random test set.
2. Imputes missing numerical values with the median.
3. One-hot encodes the categorical column.
4. Computes a few ratio features (rooms / household, bedrooms / total rooms, etc.).
5. Standardizes all numerical features.

The skeleton, with sub-pipelines per group of similar features, looks something like:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, FunctionTransformer
)
from sklearn.model_selection import train_test_split

# 1. Train/test split (do this BEFORE building the pipeline)
train_set, test_set = train_test_split(housing, test_size=0.2,
                                        random_state=42)

# 2-5. Pipeline definitions
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

ratio_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(column_ratio, feature_names_out="ratio"),
    StandardScaler(),
)

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessing = ColumnTransformer([
    ("rooms_per_house",  ratio_pipeline, ["total_rooms", "households"]),
    ("bedrooms_ratio",   ratio_pipeline, ["total_bedrooms", "total_rooms"]),
    ("cat", cat_pipeline, ["ocean_proximity"]),
], remainder=num_pipeline)

X_train_prepared = preprocessing.fit_transform(train_set)
```

Knowing the *shape* of this from memory — that you build sub-pipelines for similar features, then combine them with a `ColumnTransformer` — is the practical skill the exercise tests.

---

## 17. The lecture's joke worth remembering

> "80% of data science is cleaning the data, and 20% is complaining about cleaning the data."

The exaggeration is half the point — preprocessing is most of the work, and underestimating it is the most common reason ML projects stall.

---

## 18. Summary cheat sheet

| Step | Default choice | Watch out for |
|---|---|---|
| Train/test split | 80/20, stratified on important feature | Snooping bias if you EDA first |
| Random vs stable split | Hash on stable ID for reproducibility | Random splits drift as data grows |
| Missing numeric | `SimpleImputer(strategy="median")` | Sentinel values (0, -1, 999) |
| Missing categorical | `strategy="most_frequent"` | — |
| Categorical encoding | One-hot (`OneHotEncoder`) | Ordinal only when truly ordered |
| Long-tailed numeric | `np.log()` (in `FunctionTransformer`) | Domain issues with zeros / negatives |
| Numeric scaling | `StandardScaler` | Min-max is outlier-sensitive |
| Fit timing | `fit` on **train only** | `fit_transform` on test leaks info |
| Combining steps | `Pipeline` + `ColumnTransformer` | Match transformer to column type |

---
*Notes built 2026-05-27 from Day 2 slide deck (67 slides, AG Ch.2).*
