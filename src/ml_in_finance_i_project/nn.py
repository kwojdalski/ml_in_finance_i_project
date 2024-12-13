# %% [markdown]
# Neural Network Architecture

# Through extensive experimentation and hyperparameter tuning, I determined the following optimal network structure:

# 1. Input Layer:
#    - Activation: tanh
#    - Neurons: Matches input feature dimension

# 2. Hidden Layers (3 total):
#    - First Hidden Layer:
#      * 150 neurons with 33% dropout for regularization
#      * tanh activation
#    - Second Hidden Layer:
#      * 50 neurons
#      * ReLU activation
#    - Third Hidden Layer:
#      * 35 neurons
#      * ReLU activation

# 3. Output Layer:
#    - Single neuron (binary classification)
#    - Sigmoid activation function

import re
from pathlib import Path

import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.ta_indicators import filter_infinity_values, remove_duplicated_columns
from src.utils import ID_COLS

train_df_path = Path("./data/processed_train_df.pkl")
train_df = pd.read_pickle(train_df_path)
test_df_path = Path("./data/processed_test_df.pkl")
test_df = pd.read_pickle(test_df_path)

# %%
train_df.drop(columns=ID_COLS, inplace=True, errors="ignore")
test_df.drop(columns=ID_COLS, inplace=True, errors="ignore")

cols_to_drop = [
    col
    for col in train_df.columns
    if re.search(r"\D(?:([1-2]{1}[0-9])|([8-9]{1})\_)", str(col))
]
cols_to_drop = [col for col in cols_to_drop if not col.startswith(("RET", "VOLUME"))]

train_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
test_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# Remove duplicated columns
# %%
# Features Default selection
target = "RET"
features = train_df.columns.drop(target).tolist()

train_df, test_df, features = remove_duplicated_columns(train_df, test_df, features)
train_df, test_df, features = filter_infinity_values(
    train_df, test_df, features, target
)

x_train, x_test, y_train, y_test = train_test_split(
    train_df[features],
    train_df["RET"],
    test_size=0.25,
    random_state=0,
)


# %% Define the neural network architecture
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 50), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(50, 150), nn.Tanh(), nn.Dropout(0.33))
        self.layer3 = nn.Sequential(nn.Linear(150, 50), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(50, 35), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(35, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
