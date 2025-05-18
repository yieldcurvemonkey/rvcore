import sys

sys.path.append("../../")

import os
from datetime import datetime, timedelta
from typing import Optional

import time
import pandas as pd
import pytz
from termcolor import termcolor

from core.IRSwaps import IRSwaps, IRSwapQuery, IRSwapStructure
from core.CurveBuilding.IRSwaps.CME_IRSWAP_CURVE_QL_PARAMS import CME_IRSWAP_CURVE_QL_PARAMS

CURVE = "GBP-SONIA" 
INTERPOLATION = "log_linear"
PAR_TENORS = CME_IRSWAP_CURVE_QL_PARAMS[CURVE]["default_tenors"]
CSV_PATH = f"{os.path.join(os.getcwd(), "..", f"{CURVE}_par_rates.csv")}"

if __name__ == "__main__":

    t1 = time.time()

    irswap = IRSwaps(
        curve=CURVE,
        data_source="CME",
        ql_interpolation_algo=INTERPOLATION,
        pre_fetch_curves=True,
        error_verbose=True,
        max_njobs=-1,
    )
    
    ts_df = irswap.irswaps_timeseries_builder(
        start_date=datetime(2019, 1, 1), 
        end_date=datetime(2025, 5, 16), 
        queries=[(IRSwapQuery(tenor=t), t) for t in PAR_TENORS], 
        n_jobs=-1, 
        ignore_cache=False
    )
    ts_df[PAR_TENORS].to_csv(CSV_PATH)

    print(f"{termcolor.colored(f"Took: {time.time() - t1} sec", "blue")}")
