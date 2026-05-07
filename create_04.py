
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""
import os
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

import sys
sys.path.append('..')
from src.data.loader import M5Loader
from src.data.feature_engineering import create_base_features
from src.models.tft_model import create_tft_model

import warnings
warnings.filterwarnings("ignore")
"""))

# Cell 2: Data Loading and Dataset Creation
cells.append(nbf.v4.new_code_cell("""
loader = M5Loader(data_dir='../data')
# Use a minimal subset for CPU training to ensure fast execution
df = loader.load_merged_sample(n_items=100) 
df = create_base_features(df)

# Limit history to the last 200 days to drastically speed up CPU processing
max_days = 200
last_time_idx = (df['date'] - df['date'].min()).dt.days.max()
df['time_idx'] = (df['date'] - df['date'].min()).dt.days
df = df[df['time_idx'] > (last_time_idx - max_days)]
df['time_idx'] -= df['time_idx'].min() # Reset time_idx

# Convert categoricals to strings and target to float
cols_to_str = ['item_id', 'store_id', 'cat_id', 'dept_id', 'state_id', 
               'month', 'day_of_week', 'event_name_1', 'event_type_1']
for col in cols_to_str:
    if col in df.columns:
        df[col] = df[col].astype(str)
df['sales'] = df['sales'].astype(float)

# Fill missing sell_prices (important for M5)
df['sell_price'] = df.groupby('item_id')['sell_price'].transform(lambda x: x.ffill().bfill())
# If still some NAs (e.g. items with no prices at all), fill with 0
df['sell_price'] = df['sell_price'].fillna(0.0)

# Fill event names (they are NA when no event)
df['event_name_1'] = df['event_name_1'].fillna('none')
df['event_type_1'] = df['event_type_1'].fillna('none')

print(f"Data shape: {df.shape}")

max_prediction_length = 28
max_encoder_length = 56
training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="sales",
    group_ids=["item_id", "store_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["item_id", "store_id", "cat_id", "dept_id", "state_id"],
    time_varying_known_categoricals=["month", "day_of_week", "event_name_1", "event_type_1"],
    time_varying_known_reals=["time_idx", "sell_price"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["sales"],
    target_normalizer=GroupNormalizer(
        groups=["item_id", "store_id"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
    categorical_encoders={
        col: NaNLabelEncoder(add_nan=True) 
        for col in ["item_id", "store_id", "cat_id", "dept_id", "state_id", 
                    "month", "day_of_week", "event_name_1", "event_type_1"]
    }
)

# create validation set (predict last max_prediction_length days of training set)
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

# create dataloaders
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
"""))

# Cell 4: Create and Train Model
cells.append(nbf.v4.new_code_cell("""
# Create model
tft = create_tft_model(training)

# Logger
from lightning.pytorch.loggers import CSVLogger
logger = CSVLogger(save_dir="lightning_logs", name="tft_model")

# Trainer
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1,
    gradient_clip_val=0.1,
    enable_model_summary=True,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            mode="min"
        ),
        LearningRateMonitor(logging_interval="step")
    ],
    logger=logger
)

# Train
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
"""))

# Cell 5: Save Model
cells.append(nbf.v4.new_code_cell("""
# Save the best model
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Best model path: {best_model_path}")

# Load and save as a generic name for convenience
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
torch.save(best_tft.state_dict(), "best_tft.pth")
"""))

nb['cells'] = cells

with open('notebooks/04_tft_training.ipynb', 'w') as f:
    nbf.write(nb, f)
