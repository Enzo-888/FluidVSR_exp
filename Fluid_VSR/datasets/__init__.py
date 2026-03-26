from .ns2d import NavierStokes2DDataset
from .ERA5 import ERA5Dataset
from .Ocean import OceanDataset
from .RayleighBenard import RayleighBenardDataset
from .KolmogorovFlow import KolmogorovFlowDataset
from .ShallowWater import ShallowWaterDataset
from .ERA5V import ERA5VDataset

_dataset_dict = {
    "NavierStokes2D": NavierStokes2DDataset,
    "ERA5": ERA5Dataset,
    "Ocean": OceanDataset,
    "RayleighBenard": RayleighBenardDataset,
    "KolmogorovFlow": KolmogorovFlowDataset,
    "ShallowWater": ShallowWaterDataset,
    "ERA5V": ERA5VDataset,
}
