from core.Products.CurveBuilding.Cash.ParCurveModel.GeneralSpliner import GeneralSpliner
from core.Products.CurveBuilding.Cash.ParCurveModel.JPM2024 import JPM2024
from core.Products.CurveBuilding.Cash.ParCurveModel.GSW2006 import GSW2006

PAR_CURVE_MODELS = {
    "general_spliner": GeneralSpliner,
    # "jpm_2024": JPM2024, # broken 
    # "gsw_2006": GSW2006, # broken
}