"""
Advanced augmentation package for deepfake detection
Includes GAN-based face swaps and domain randomization
"""

from backend.gan_augmentation import StyleGAN2Augmenter, AdvancedAugmentationPipeline
from backend.domain_randomization import DomainRandomizer

__all__ = [
    'StyleGAN2Augmenter',
    'AdvancedAugmentationPipeline',
    'DomainRandomizer'
]