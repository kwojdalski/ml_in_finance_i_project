# %%
import logging as log
import re
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ta_indicators import filter_infinity_values, remove_duplicated_columns
from src.utils import ID_COLS

target = "RET"
train_df_path = Path("./data/processed_train_df.pkl")
test_df_path = Path("./data/processed_test_df.pkl")
if train_df_path.exists() and test_df_path.exists():
    log.info(f"Loading processed training data from {train_df_path}")
    train_df = pd.read_pickle(train_df_path)
    test_df = pd.read_pickle(test_df_path)
else:
    log.info(f"Saving processed training data to {train_df_path}")
    train_df.to_pickle(train_df_path)
    test_df.to_pickle(test_df_path)

# %% [markdown]
# #### Columns to drop
# They could bring in some predictive power, but we don't want to use them in this case
# as the scope is limited for this project
# ['ID', 'STOCK', 'DATE', 'INDUSTRY', 'INDUSTRY_GROUP', 'SECTOR', 'SUB_INDUSTRY']
# %%
train_df.drop(columns=ID_COLS, inplace=True, errors="ignore")
test_df.drop(columns=ID_COLS, inplace=True, errors="ignore")

# %% [markdown]
# Assumption is that probably some technical indicators are not useful for the prediction.
# For instance SMA(10), SMA(11) etc. dont give any information in the context of RET.
# It's an arbitrary choice, but we want to keep the number of features low

# %% [markdown]
# #### Further data wrangling
# %%
cols_to_drop = [
    col
    for col in train_df.columns
    if re.search(
        r"\D(?:([1-2]{1}[0-9])|([8-9]{1})\_)",
        str(col),
    )
    and not col.startswith(("RET", "VOLUME"))  # don't drop RET and VOLUME
]

train_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
# %%
# Define features before using them
features = [col for col in train_df.columns if col != target]

# %% Filter out infinity values
train_df, test_df, features = filter_infinity_values(
    train_df, test_df, features, target
)

# Remove duplicated columns
train_df, test_df, features = remove_duplicated_columns(train_df, test_df, features)

# %%
# Features
# Default selection
features = train_df.columns.drop(target).tolist()

# %% [markdown]
# ## ML DecisionTreeClassifier

# %%
# Train and test set splitting
x_train, x_test, y_train, y_test = train_test_split(
    train_df[features],
    train_df["RET"],
    test_size=0.25,
    random_state=0,
)

# %%
# Logistic Regression
# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# max_iter controls the maximum number of iterations for the solver to converge
# Increased from default 100 to handle complex datasets that need more iterations
lr = LogisticRegression(max_iter=10000)


# learn
lr = LogisticRegression().fit(x_train_scaled, y_train)
# test
y_lr_predict = lr.predict(x_test_scaled)
print(classification_report(y_test, y_lr_predict, digits=3))
# %%

# %%
