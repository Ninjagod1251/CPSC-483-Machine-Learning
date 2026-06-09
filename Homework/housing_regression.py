    """
    CPSC 483 - Homework 1, Question 7
    Develop and evaluate a regression model to predict median_house_value.

    Follows the conventions in 02_end_to_end_machine_learning_project.ipynb
    (Hands-On Machine Learning, Geron).
    """

    from pathlib import Path
    import urllib.request

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error


    # ---------------------------------------------------------------------------
    # Step 1: Download the housing dataset used in the textbook
    # ---------------------------------------------------------------------------
    url = "https://raw.githubusercontent.com/ageron/data/main/housing/housing.csv"
    Path("datasets").mkdir(exist_ok=True)
    csv_path = Path("datasets/housing.csv")
    if not csv_path.is_file():
        urllib.request.urlretrieve(url, csv_path)
    housing = pd.read_csv(csv_path)

    # Step 2: The target variable
    TARGET = "median_house_value"
    print("Target variable:", TARGET)

    # ---------------------------------------------------------------------------
    # Step 3: Scatter plot of median_income vs median_house_value
    # ---------------------------------------------------------------------------
    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                alpha=0.1, grid=True)
    plt.title("median_income vs median_house_value")
    plt.show()

    # ---------------------------------------------------------------------------
    # Step 4: Remove all rows where median_house_value >= 500,000
    # ---------------------------------------------------------------------------
    housing = housing[housing["median_house_value"] < 500_000].reset_index(drop=True)
    print("Rows after removing capped values:", len(housing))

    # ---------------------------------------------------------------------------
    # Step 5: Train/test split, 20% test, random sampling
    # ---------------------------------------------------------------------------
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    X_train = train_set.drop(columns=[TARGET])
    y_train = train_set[TARGET].copy()
    X_test = test_set.drop(columns=[TARGET])
    y_test = test_set[TARGET].copy()

    # ---------------------------------------------------------------------------
    # Step 6: New variable total_bedrooms/total_rooms (on train) + correlation
    # ---------------------------------------------------------------------------
    train_explore = train_set.copy()
    train_explore["bedrooms_ratio"] = (
        train_explore["total_bedrooms"] / train_explore["total_rooms"]
    )
    corr = train_explore["bedrooms_ratio"].corr(train_explore["median_house_value"])
    print(f"Correlation of bedrooms_ratio with median_house_value: {corr:.4f}")

    # ---------------------------------------------------------------------------
    # Step 7: Pre-processing pipeline
    # ---------------------------------------------------------------------------
    num_attribs = ["housing_median_age", "total_rooms", "total_bedrooms",
                "population", "households", "median_income"]
    cat_attribs = ["ocean_proximity"]


    def column_ratio(X):
        return X[:, [0]] / X[:, [1]]


    # numerical: median impute -> log transform -> scale to 0-1
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        MinMaxScaler(),
    )

    # categorical: most-frequent impute -> one-hot encode
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    # new variable: total_bedrooms / total_rooms -> scale to 0-1
    ratio_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=lambda f, n: ["bedrooms_ratio"]),
        MinMaxScaler(),
    )

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
        ("ratio", ratio_pipeline, ["total_bedrooms", "total_rooms"]),
    ])

    # How many variables are in the pre-processed training dataset?
    X_train_prepared = preprocessing.fit_transform(X_train)
    print("Number of variables after preprocessing:", X_train_prepared.shape[1])

    # ---------------------------------------------------------------------------
    # Step 8: Add LinearRegression, 3-fold CV RMSE on the training set
    # ---------------------------------------------------------------------------
    lin_reg = make_pipeline(preprocessing, LinearRegression())

    neg_mse = cross_val_score(lin_reg, X_train, y_train,
                            scoring="neg_mean_squared_error", cv=3)
    cv_rmse = np.sqrt(-neg_mse)
    print(f"3-fold CV RMSE (train): mean = {cv_rmse.mean():.2f}, folds = {cv_rmse.round(2)}")

    # ---------------------------------------------------------------------------
    # Step 9: RMSE on the test dataset
    # ---------------------------------------------------------------------------
    lin_reg.fit(X_train, y_train)
    test_predictions = lin_reg.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f"Test RMSE: {test_rmse:.2f}")
