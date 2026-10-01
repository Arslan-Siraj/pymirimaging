from enum import Enum


class m2Normalization(str, Enum):
    NONE = "None"
    TIC = "TIC"
    Sum = "Sum"
    Mean = "Mean"
    Max = "Max"
    RMS = "RMS"


class m2Pooling(str, Enum):
    NONE = "None"
    Mean = "Mean"
    Median = "Median"
    Maximum = "Maximum"
    Sum = "Sum"


class m2ImageNormalization(str, Enum):
    NONE = "None"
    MinMax = "MinMax"


class m2SpectralSelection(str, Enum):
    Nearest = "Nearest"
    Exact = "Exact"
    Window = "Window"


m2NormalizationNone = m2Normalization.NONE
m2NormalizationTIC = m2Normalization.TIC
m2NormalizationSum = m2Normalization.Sum
m2NormalizationMean = m2Normalization.Mean
m2NormalizationMax = m2Normalization.Max
m2NormalizationRMS = m2Normalization.RMS

m2PoolingNone = m2Pooling.NONE
m2PoolingMean = m2Pooling.Mean
m2PoolingMedian = m2Pooling.Median
m2PoolingMaximum = m2Pooling.Maximum
m2PoolingSum = m2Pooling.Sum
