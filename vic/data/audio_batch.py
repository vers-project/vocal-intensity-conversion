"""AudioBatch — re-exported from ``audio_utils.data.audio_batch``.

Kept as a module so the ``from vic.data.audio_batch import AudioBatch`` imports
across the package keep working; there is no VIC-specific implementation.
"""
from audio_utils.data.audio_batch import AudioBatch

__all__ = ["AudioBatch"]
