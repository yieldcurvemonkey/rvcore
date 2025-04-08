from typing import Dict, TypeAlias, Literal
import pandas as pd

StrikeOffset: TypeAlias = float
VolGrid: TypeAlias = pd.DataFrame
SCube: TypeAlias = Dict[StrikeOffset, VolGrid]

SABRParams: TypeAlias = Dict[Literal["alpha", "beta", "nu", "rho"], float]