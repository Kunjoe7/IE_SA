import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchtuples as tt

from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv

# Set random seeds for reproducibility
np.random.seed(1234)
_ = torch.manual_seed(123)

# Load the preprocessed data
df = pd.read_csv('/project/lwang/xqin5/SAcompare/NWTCO/nwtco.csv')
data = df

# Split features and target
X = data.drop(columns=["edrel", "rel"])
y_time = data["edrel"]
y_event = data["rel"]

# Split the data into training and test sets
X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
    X, y_time, y_event, test_size=0.25, random_state=42
)

# Merge data to create complete training and test datasets
train_df = X_train.copy()
train_df['duration'] = y_time_train
train_df['event'] = y_event_train

test_df = X_test.copy()
test_df['duration'] = y_time_test
test_df['event'] = y_event_test

# Split the training dataset into training and validation sets
df_val = train_df.sample(frac=0.2, random_state=42)
df_train = train_df.drop(df_val.index)

# Extract features and target data (no additional standardization)
x_train = df_train.drop(columns=['duration', 'event']).astype('float32').values
x_val = df_val.drop(columns=['duration', 'event']).astype('float32').values
x_test = test_df.drop(columns=['duration', 'event']).astype('float32').values

# Extract target data (duration and event status)
y_train = (df_train['duration'].values.astype('float32'), df_train['event'].values.astype('float32'))
y_val = (df_val['duration'].values.astype('float32'), df_val['event'].values.astype('float32'))
durations_test, events_test = (test_df['duration'].values.astype('float32'), test_df['event'].values.astype('float32'))
val = (x_val, y_val)

# Define model parameters
in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

# Create model network
net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)

# Define the CoxTime model
model = CoxTime(net, tt.optim.Adam)

# Find the best learning rate
batch_size = 256
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
best_lr = lrfinder.get_best_lr()
print(f"Best learning rate: {best_lr}")

# Set the learning rate and train the model
model.optimizer.set_lr(0.01)
epochs = 64
callbacks = [tt.callbacks.EarlyStopping()]
verbose = False  # Disable verbose output for cluster environment

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val, val_batch_size=batch_size)

# Compute partial log likelihood
partial_log_likelihood = model.partial_log_likelihood(*val).mean()
print(f"Partial log likelihood: {partial_log_likelihood}")

# Compute baseline hazards
model.compute_baseline_hazards()

# Predict survival probabilities
surv_train = model.predict_surv_df(x_train)
surv_val = model.predict_surv_df(x_val)
surv_test = model.predict_surv_df(x_test)

# Evaluate model performance on training, validation, and test sets
ev_train = EvalSurv(surv_train, y_train[0], y_train[1], censor_surv='km')
c_index_train = ev_train.concordance_td()
print(f"C-index on training set: {c_index_train}")

ev_val = EvalSurv(surv_val, y_val[0], y_val[1], censor_surv='km')
c_index_val = ev_val.concordance_td()
print(f"C-index on validation set: {c_index_val}")

ev_test = EvalSurv(surv_test, durations_test, events_test, censor_surv='km')
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
