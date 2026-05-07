import os
import sys
import logging
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Monitors data distribution for drift using Population Stability Index (PSI).
    
    PSI < 0.1 = No significant change
    PSI > 0.1 = Minor drift (warning)
    PSI > 0.2 = Major drift (trigger retraining)
    """
    
    def __init__(self, baseline_df: pd.DataFrame, threshold_warning=0.1, threshold_retrain=0.2):
        self.baseline_df = baseline_df
        self.threshold_warning = threshold_warning
        self.threshold_retrain = threshold_retrain
        logger.info("DriftDetector initialized with baseline data.")

    def compute_psi(self, current_df: pd.DataFrame, feature: str, n_bins: int = 10) -> float:
        """
        Compute PSI between baseline and current distribution for a specific feature.
        Formula: PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        if feature not in self.baseline_df.columns or feature not in current_df.columns:
            logger.warning(f"Feature {feature} not found in both dataframes.")
            return 0.0

        # Create bins based on baseline data
        baseline_values = self.baseline_df[feature].dropna()
        current_values = current_df[feature].dropna()

        if len(baseline_values) == 0 or len(current_values) == 0:
            return 0.0

        # Calculate bin boundaries
        bins = np.histogram_bin_edges(baseline_values, bins=n_bins)
        
        # Calculate frequency distributions
        expected_percents = np.histogram(baseline_values, bins=bins)[0] / len(baseline_values)
        actual_percents = np.histogram(current_values, bins=bins)[0] / len(current_values)

        # Handle zero frequencies by adding a small epsilon (Laplace smoothing)
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        # Compute PSI
        psi_score = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return float(psi_score)

    def check_drift(self, current_df: pd.DataFrame) -> dict:
        """
        Check all numeric features for drift.
        Returns dict with {feature: psi_score, 'action': 'none'|'warn'|'retrain'}
        """
        numeric_features = self.baseline_df.select_dtypes(include=[np.number]).columns
        drift_results = {}
        max_psi = 0.0

        for feature in numeric_features:
            # Skip target column and indices
            if feature in ['sales', 'time_idx', 'wm_yr_wk']:
                continue
                
            psi = self.compute_psi(current_df, feature)
            drift_results[feature] = round(psi, 4)
            max_psi = max(max_psi, psi)

        # Determine action
        if max_psi >= self.threshold_retrain:
            action = "retrain"
        elif max_psi >= self.threshold_warning:
            action = "warn"
        else:
            action = "none"

        return {
            "feature_psi": drift_results,
            "max_psi": round(max_psi, 4),
            "action": action
        }

    def check_performance_drift(self, current_wmape: float, baseline_wmape: float) -> bool:
        """
        Trigger retraining if WMAPE degraded > 5% relative.
        Example: Baseline 20%, Current 21.5% -> 7.5% degradation -> Trigger True.
        """
        if baseline_wmape == 0:
            return False
            
        relative_degradation = (current_wmape - baseline_wmape) / baseline_wmape
        
        if relative_degradation > 0.05:
            logger.warning(f"Performance drift detected: WMAPE degraded by {relative_degradation:.2%}")
            return True
        
        return False

if __name__ == "__main__":
    # Quick test logic
    baseline = pd.DataFrame({'price': np.random.normal(10, 1, 1000)})
    current = pd.DataFrame({'price': np.random.normal(10.5, 1, 1000)}) # Shifted
    
    detector = DriftDetector(baseline)
    results = detector.check_drift(current)
    print(f"Drift Check Results: {results}")
