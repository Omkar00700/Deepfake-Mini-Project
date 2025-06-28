"""
Advanced training techniques for deepfake detection
Includes curriculum learning, mixup, and learning rate scheduling
"""

from backend.curriculum_learning import CurriculumSampler, CurriculumDataGenerator, CurriculumCallback
from backend.mixup import Mixup, ManifoldMixup, MixupLayer, MixupLoss, MixupCallback
from backend.lr_scheduling import LayerwiseLRDecay, OneCycleLR, CosineAnnealingWarmRestarts, AdamWOptimizer

__all__ = [
    'CurriculumSampler',
    'CurriculumDataGenerator',
    'CurriculumCallback',
    'Mixup',
    'ManifoldMixup',
    'MixupLayer',
    'MixupLoss',
    'MixupCallback',
    'LayerwiseLRDecay',
    'OneCycleLR',
    'CosineAnnealingWarmRestarts',
    'AdamWOptimizer'
]