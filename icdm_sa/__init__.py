"""
ICDM-SA: Interactive and Explainable Survival Analysis

A Python package for interactive and explainable survival analysis with multiple datasets.
"""

__version__ = '0.1.0'
__author__ = 'ICDM-SA Team'

# Import main components
from .algorithm import (
    MultiTaskModel,
    EGTrainer,
    NoRegularizationTrainer,
    NoRegularizationTrainer1,
    Cindex,
    MultiTaskDataset,
    TabTransformerMultiTaskModel,
    brier_score,
    unique_value_counts
)

# Import dataset-specific models
from .datasets.flchain.flchain_model import FLCHAINModel

__all__ = [
    'MultiTaskModel',
    'EGTrainer', 
    'NoRegularizationTrainer',
    'NoRegularizationTrainer1',
    'Cindex',
    'MultiTaskDataset',
    'TabTransformerMultiTaskModel',
    'brier_score',
    'unique_value_counts',
    'FLCHAINModel'
]