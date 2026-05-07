import os
import sys
import json
import logging
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

from src.data.loader import M5Loader
from src.data.feature_engineering import create_base_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_retraining(
    store_filter: list = None,
    max_epochs: int = 10,  # Lowered for automated retraining speed
    raw_dir: str = 'data',
    checkpoint_dir: str = 'models/tft',
) -> str:
    """
    Full retraining pipeline:
    1. Load latest raw data
    2. Run feature engineering
    3. Build TimeSeriesDataSet
    4. Train TFT with early stopping
    5. Evaluate and Save
    """
    logger.info("Starting automated retraining pipeline...")

    # 1. Load data
    loader = M5Loader(data_dir=raw_dir)
    # For retraining, we typically load a larger sample or the full set
    # Here we simulate with a 200-item sample for the pipeline demonstration
    data = loader.load_merged_sample(n_items=200)
    
    if store_filter:
        data = data[data['store_id'].isin(store_filter)]

    # 2. Feature Engineering
    df = create_base_features(data)
    
    # Create time_idx
    df['time_idx'] = df['d'].str.extract('(\\d+)').astype(int)
    df['time_idx'] -= df['time_idx'].min()
    
    # Ensure types and handle NaNs
    df['sales'] = df['sales'].astype(float)
    
    # Fill missing real features (sell_price can be NA before item intro)
    for col in ["sell_price", "day_of_week", "month", "sales"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    # 3. Build TimeSeriesDataSet
    max_prediction_length = 28
    max_encoder_length = 56
    training_cutoff = df["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["item_id", "dept_id", "store_id"],
        time_varying_known_reals=["sell_price", "day_of_week", "month"],
        time_varying_unknown_reals=["sales"],
        target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    
    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

    # 4. Train TFT
    pl.seed_everything(42)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min"),
            ModelCheckpoint(dirpath=checkpoint_dir, filename="retrained_tft-{epoch:02d}", save_top_k=1, monitor="val_loss", mode="min"),
        ],
        logger=TensorBoardLogger(save_dir="lightning_logs", name="retraining"),
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    logger.info(f"Training on {len(training)} samples...")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # 5. Evaluate (Simplistic check)
    best_model_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Retraining complete. Best model saved at: {best_model_path}")
    
    # Save metadata
    metrics = {
        "val_loss": float(trainer.callback_metrics["val_loss"]),
        "status": "success",
        "checkpoint": best_model_path
    }
    with open(os.path.join(checkpoint_dir, "latest_metrics.json"), 'w') as f:
        json.dump(metrics, f)

    return best_model_path

if __name__ == "__main__":
    # Ensure dirs exist
    os.makedirs('models/tft', exist_ok=True)
    # Note: This will only run if data is present
    try:
        path = run_retraining(max_epochs=1)
        print(f"Pipeline Test Success. Checkpoint: {path}")
    except Exception as e:
        logger.error(f"Pipeline Test Failed: {e}")
