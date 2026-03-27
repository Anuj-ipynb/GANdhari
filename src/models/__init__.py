# filename: src/models/__init__.py
from .generator import UNetGenerator
from .discriminator import PatchGAN

__all__ = ['UNetGenerator', 'PatchGAN']