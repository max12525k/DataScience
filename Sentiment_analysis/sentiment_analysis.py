# %%
# import all the necessary packages
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from nltk.tokenize import word_tokenize
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor


def load_data(file_name):
    """
    Reads the data from the file and returns a pandas dataframe.
    """
    return pd.read_csv(file_name)


def train_test_df(df, split_raito=0.2):
    """
    Splits the dataframe into train and test dataframes.
    """
    n_splits = int(len(df) * split_raito)
    df_train = df[:n_splits]
    df_test = df[n_splits:]
    return df_train, df_test


def preprocess_date(df, date_col="PublishDate"):
    """
    Generate basic date time features
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday
    df["hour"] = df[date_col].dt.hour
    df["minute"] = df[date_col].dt.minute
    df.drop(columns=date_col, inplace=True)
    return df


def score_weighted_mse(y_pred, y):
    """
    Calculates the weighted mean absolute error for 2 target variables.
    """
    mse_title = mean_absolute_error(y[:, 0], y_pred[:, 0])
    mse_headline = mean_absolute_error(y[:, 1], y_pred[:, 1])

    return max(0, 1 - (0.4 * mse_title + 0.6 * mse_headline))


def cross_val_weighted_mse(estimator, X, y, cv=5):
    """
    Calculates the weighted mean absolute error  for Kfold cross validaiton of a model.
    """
    scores = []
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx, :], y[test_idx, :]

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        scores.append(score_weighted_mse(y_pred, y_test))
    return scores


def column_transform(df, cat_cols, num_cols, text_cols, date_time_feature_cols):
    """
    Transforms the appropriate columns with appropriate tranformations.
    """
    # apply one hot encoding to categorical columns (cat_ohe)
    # apply min max scaling to numerical columns (num_scaleer)
    # apply one hot encoding to date time features (date_time_endcoder)
    # apply tfidf vectorizer to Title and Headline while using word tokenizer from nltk
    # ( text_title_tfidf, text_headline_tfidf)
    # apply count vectorizer to Title and Headline while using word tokenizer from nltk
    # (text_title_countVect, text_headline_countVect)
    # drop other columns

    pipe = ColumnTransformer(
        [
            (
                "cat_ohe",
                OneHotEncoder(categories="auto", sparse=True, handle_unknown="ignore"),
                cat_cols,
            ),
            ("num_scaler", MinMaxScaler(), num_cols),
            (
                "text_title_countVect",
                CountVectorizer(tokenizer=word_tokenize, token_pattern=None),
                text_cols[0],
            ),
            (
                "text_title_tfidf",
                TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None),
                text_cols[0],
            ),
            (
                "text_headline_countVect",
                CountVectorizer(tokenizer=word_tokenize, token_pattern=None),
                text_cols[1],
            ),
            (
                "text_headline_tfidf",
                TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None),
                text_cols[1],
            ),
            (
                "date_time_encoder",
                OneHotEncoder(handle_unknown="ignore"),
                date_time_feature_cols,
            ),
        ],
        remainder="drop",
    )
    pipe.fit(df)
    return pipe


def apply_base_preprocess(df, date_col="PublishDate"):
    """
    Fill nan values with unknown string, set id column as index
    and get date time features
    """
    df["Source"] = df["Source"].fillna("Unknown")
    df = df.set_index("IDLink")
    df = preprocess_date(df, date_col)
    return df


# %%

# %%
# define categorical and numerical columns and text columns
# target columns and date time features columns

cat_cols = ["Source", "Topic"]
num_cols = ["Facebook", "GooglePlus", "LinkedIn"]
target = ["SentimentTitle", "SentimentHeadline"]
text_cols = ["Title", "Headline"]
date_time_feature_cols = ["year", "month", "day", "weekday", "hour", "minute"]

# Actual code for generating prediction on test set provided
# load train and test data
df = load_data("train_file.csv")
df_test = load_data("test_file.csv")
# save the  id column for test set
index = df_test["IDLink"]
# apply  basic preprocessing on train and test data
df = apply_base_preprocess(df)
df_test = apply_base_preprocess(df_test)
# get target values and drop target from orignal dataframe as it will not be present in test set
y = df[target].to_numpy()
df = df.drop(columns=target)

# fit the column transformer on train data
trans = column_transform(
    df,
    cat_cols=cat_cols,
    num_cols=num_cols,
    text_cols=text_cols,
    date_time_feature_cols=date_time_feature_cols,
)
# tranfrom train and test data
X = trans.transform(df)
X_test = trans.transform(df_test)
#

# intialize multioutput model with xgboost regressor model
reg = MultiOutputRegressor(XGBRegressor(n_estimators=1138, max_depth=3))
# fit the model on training data
reg.fit(X, y)

# predict on the test set provided
y_pred = reg.predict(X_test)

# combine id and prediction
df_pred = pd.DataFrame(y_pred, index=index, columns=target)
# save the prediction to csv file
df_pred.to_csv("submission.csv")
