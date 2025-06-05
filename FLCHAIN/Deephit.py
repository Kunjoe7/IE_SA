import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchtuples as tt

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

# Set random seeds for reproducibility
np.random.seed(1234)
_ = torch.manual_seed(123)

# Load the data
df = pd.read_csv('/project/lwang/xqin5/SAcompare/FLCHAIN/flchain.csv')
data = df

# data = data[data["survival_time"] <= 1500]
# Load the preprocessed data
X = data.drop(columns=["duration", "event"])
y_time = data["duration"]
y_event = data["event"]

# Split the data into training, validation, and test sets
X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
    X, y_time, y_event, test_size=0.2, random_state=42)

X_train, X_val, y_time_train, y_time_val, y_event_train, y_event_val = train_test_split(
    X_train, y_time_train, y_event_train, test_size=0.2, random_state=42)

# Create DataFrames for training, validation, and test sets
df_train = X_train.copy()
df_train['duration'] = y_time_train
df_train['event'] = y_event_train

df_val = X_val.copy()
df_val['duration'] = y_time_val
df_val['event'] = y_event_val

test_df = X_test.copy()
test_df['duration'] = y_time_test
test_df['event'] = y_event_test

# Extract features and target data
x_train = df_train.drop(columns=['duration', 'event']).astype('float32').values
x_val = df_val.drop(columns=['duration', 'event']).astype('float32').values
x_test = test_df.drop(columns=['duration', 'event']).astype('float32').values

# Discretize survival time using the same bins for all datasets
num_intervals = 20
time_bins = np.linspace(y_time.min(), y_time.max(), num_intervals)

y_train_discrete = np.digitize(df_train['duration'], bins=time_bins) - 1
y_val_discrete = np.digitize(df_val['duration'], bins=time_bins) - 1
y_test_discrete = np.digitize(test_df['duration'], bins=time_bins) - 1

# Create target variables by combining event status and discretized survival time
y_train = (y_train_discrete.astype('int64'), df_train['event'].values.astype('float32'))
y_val = (y_val_discrete.astype('int64'), df_val['event'].values.astype('float32'))
y_test = (y_test_discrete.astype('int64'), test_df['event'].values.astype('float32'))
durations_test, events_test = test_df['duration'].values.astype('float32'), test_df['event'].values.astype('float32')

in_features = x_train.shape[1]
num_nodes = [32, 32]
num_risks = 1
out_features = 4000 

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm=True, dropout=0.01)
model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1)

# Find the best learning rate
batch_size = 64
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
best_lr = lrfinder.get_best_lr()
print(f"Best learning rate: {best_lr}")

# Set the learning rate and train the model
model.optimizer.set_lr(0.1)
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = False  # Disable verbose output for cluster environment

# Predict survival probabilities for each dataset
surv_train = model.predict_surv(x_train)
surv_val = model.predict_surv(x_val)
surv_test = model.predict_surv(x_test)

# Convert survival probabilities to DataFrame format for easy processing
surv_train_df = model.predict_surv_df(x_train)
surv_val_df = model.predict_surv_df(x_val)
surv_test_df = model.predict_surv_df(x_test)

# Function to manually compute the negative log-likelihood
def compute_negative_log_likelihood(surv_df, durations, events):
    nll = 0.0
    eps = 1e-10  # Small constant to avoid log(0)

    # Iterate over each sample to find the event occurrence time and calculate negative log-likelihood
    for i in range(len(durations)):
        duration = int(durations[i])  # Current sample's survival duration
        event = events[i]             # Current sample's event status (1 if event occurred, 0 if censored)

        if event == 1:
            # If the event occurred, get the survival probability at the time of event occurrence
            survival_prob = surv_df.iloc[duration, i] if duration < len(surv_df) else eps
            nll -= torch.log(torch.tensor(survival_prob + eps)).item()  # Take log and add to the total sum
    
    # Return the average negative log-likelihood
    return nll / len(durations)

# Calculate negative log-likelihood for training, validation, and test sets
nll_train = compute_negative_log_likelihood(surv_train_df, y_train[0], y_train[1])
nll_val = compute_negative_log_likelihood(surv_val_df, y_val[0], y_val[1])
nll_test = compute_negative_log_likelihood(surv_test_df, y_test[0], y_test[1])

# Print the negative log-likelihood for each dataset
print(f"Negative Log-Likelihood on training set: {nll_train}")
print(f"Negative Log-Likelihood on validation set: {nll_val}")
print(f"Negative Log-Likelihood on test set: {nll_test}")

# Predict survival probabilities
surv_train = model.predict_surv_df(x_train)
surv_val = model.predict_surv_df(x_val)
surv_test = model.predict_surv_df(x_test)

# Evaluate model performance on training, validation, and test sets using discrete times
ev_train = EvalSurv(surv_train, y_train[0], y_train[1], censor_surv='km')
c_index_train = ev_train.concordance_td()
print(f"C-index on training set: {c_index_train}")

ev_val = EvalSurv(surv_val, y_val[0], y_val[1], censor_surv='km')
c_index_val = ev_val.concordance_td()
print(f"C-index on validation set: {c_index_val}")

ev_test = EvalSurv(surv_test, y_test[0], y_test[1], censor_surv='km')
c_index_test = ev_test.concordance_td()
print(f"C-index on test set: {c_index_test}")

# Compute Brier Score, integrated Brier Score, and integrated negative log-likelihood on the test set
time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)

# Brier Score and Integrated Brier Score
brier_score = ev_test.brier_score(time_grid)
integrated_brier_score = ev_test.integrated_brier_score(time_grid)
print(f"Integrated Brier Score (test set): {integrated_brier_score}")

# Integrated Negative Log-Likelihood
integrated_nbll = ev_test.integrated_nbll(time_grid)
print(f"Integrated negative log-likelihood (test set): {integrated_nbll}")
