import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class M5Loader:
    """
    Loader for the M5 Forecasting dataset.
    """
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.calendar_path = os.path.join(data_dir, 'calendar.csv')
        self.sales_path = os.path.join(data_dir, 'sales_train_evaluation.csv')
        self.prices_path = os.path.join(data_dir, 'sell_prices.csv')

    def load_calendar(self):
        logger.info("Loading calendar data...")
        df = pd.read_csv(self.calendar_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def load_sales(self):
        logger.info("Loading sales data...")
        return pd.read_csv(self.sales_path)

    def load_prices(self):
        logger.info("Loading prices data...")
        return pd.read_csv(self.prices_path)

    def load_all(self):
        """Loads calendar, sales, and prices datasets."""
        logger.info("Loading all M5 datasets...")
        return self.load_calendar(), self.load_sales(), self.load_prices()

    def load_merged_sample(self, n_items=100):
        """
        Loads a sample of items and merges them with calendar and prices.
        Useful for EDA without loading all 80M rows.
        """
        logger.info(f"Loading merged sample of {n_items} items...")
        sales = self.load_sales().sample(n_items, random_state=42)
        calendar = self.load_calendar()
        prices = self.load_prices()

        # Melt sales to long format
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        sales_long = sales.melt(id_vars=id_cols, var_name='d', value_name='sales')
        
        # Merge with calendar
        df = sales_long.merge(calendar, on='d', how='left')
        
        # Merge with prices
        df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        
        return df

if __name__ == "__main__":
    loader = M5Loader()
    cal = loader.load_calendar()
    print(f"Calendar shape: {cal.shape}")
