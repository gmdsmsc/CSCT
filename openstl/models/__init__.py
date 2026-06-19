#from .TSST_PredFormer import PredFormer as InteractivePredictionModel
from .TSST_conditioning_only import InteractivePredictionModel as InteractivePredictionModel
#from .TSST_attention_only import InteractivePredictionModel as InteractivePredictionModel
#from .TSST_final import InteractivePredictionModel

__all__ = [
    'InteractivePredictionModel',
]