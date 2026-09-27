from vic.models.blocks import (
    ConvBackbone,
    ConvEncoder,
    ConvLayerConfig,
    FiLM,
    ScalarIntensityEmbedding,
    SinusoidalIntensityEmbedding,
    SequenceBackbone,
    TransformerEncoder,
    masked_mean_pool,
)
from vic.models.converter import ContextualConverter, ConvConverter, FrameConverter
from vic.models.discriminator import (
    ConvTrueFakeDiscriminator,
    PooledDiscriminator,
    TrueFakeDiscriminator,
)
from vic.models.predictor import (
    ConvIntensityPredictor,
    MLPIntensityPredictor,
    TransformerIntensityPredictor,
    build_predictor,
)

__all__ = [
    # Embedders
    "ScalarIntensityEmbedding",
    "SinusoidalIntensityEmbedding",
    # Building blocks
    "SequenceBackbone",
    "TransformerEncoder",
    "ConvBackbone",
    "ConvEncoder",
    "ConvLayerConfig",
    "FiLM",
    "masked_mean_pool",
    # Models
    "TransformerIntensityPredictor",
    "MLPIntensityPredictor",
    "ConvIntensityPredictor",
    "build_predictor",
    "FrameConverter",
    "ContextualConverter",
    "ConvConverter",
    "PooledDiscriminator",
    "TrueFakeDiscriminator",
    "ConvTrueFakeDiscriminator",
]
