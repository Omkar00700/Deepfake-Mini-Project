"""
Advanced evaluation techniques for deepfake detection
Includes stratified k-fold, Bayesian model averaging, and calibration
"""

from backend.stratified_kfold import StratifiedGroupKFoldCV, DeepfakeDatasetSplitter
from backend.bayesian_ensemble import DeepEnsemble, BayesianModelAveraging, PostProcessingCalibrator
from backend.calibration import TemperatureScaling, IsotonicCalibration, CombinedCalibration, plot_calibration_curve

__all__ = [
    'StratifiedGroupKFoldCV',
    'DeepfakeDatasetSplitter',
    'DeepEnsemble',
    'BayesianModelAveraging',
    'PostProcessingCalibrator',
    'TemperatureScaling',
    'IsotonicCalibration',
    'CombinedCalibration',
    'plot_calibration_curve'
]