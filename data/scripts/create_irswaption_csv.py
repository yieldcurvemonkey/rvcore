import sys

sys.path.append("../../")

import os
from datetime import datetime, timedelta
from typing import Optional

import tqdm
import QuantLib as ql
import time
import pandas as pd
import pytz
from termcolor import termcolor

from core.IRSwaps import IRSwaps
from core.IRSwaptions import IRSwaptions
from core.CurveBuilding.IRSwaps.CME_IRSWAP_CURVE_QL_PARAMS import CME_IRSWAP_CURVE_QL_PARAMS
from core.utils.ql_utils import datetime_to_ql_date

CURVE = "USD-SOFR-1D"
INTERPOLATION = "log_linear"
CSV_PATH = f"{os.path.join(os.getcwd(), "..", "usd_vol_cube.csv")}"
DATE_COL = "Date"

expiries = ["1M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "30Y"]
tails = ["1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "30Y"]
strike_offsets_bps = [-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200]

eastern = "US/Eastern"
chi_now = datetime.now(tz=pytz.timezone("US/Central"))
chi_today = datetime(year=chi_now.year, month=chi_now.month, day=chi_now.day)

if __name__ == "__main__":

    t1 = time.time()

    usd_ois = IRSwaps(
        curve=CURVE,
        data_source="CME",
        ql_interpolation_algo=INTERPOLATION,
        error_verbose=True,
        max_njobs=-1,
    )
    usd_swaptions = IRSwaptions(
        data_source="GITHUB_USD_MONKEY_CUBE",
        irswaps_product=usd_ois,
        error_verbose=True,
        max_n_jobs=-1,
    )

    curr_df = pd.read_csv(CSV_PATH)
    curr_df[DATE_COL] = pd.to_datetime(curr_df[DATE_COL])
    latest_in_csv = curr_df[DATE_COL].iloc[-1]
    
    ql_cubes = usd_swaptions.fetch_qlcubes(
        start_date=datetime(latest_in_csv.year, latest_in_csv.month, latest_in_csv.day), end_date=chi_today, to_pydt=True
    )

    records = []
    for d, cube in tqdm.tqdm(ql_cubes.items(), desc="FLATTENING CUBE..."):
        cube.enableExtrapolation()

        ql.Settings.instance().evaluationDate = datetime_to_ql_date(d)
        row: dict[str, float | datetime] = {"Date": d}
        for e in expiries:
            for t in tails:
                smile: ql.SmileSection = cube.smileSection(ql.Period(e), ql.Period(t), True)
                atm = smile.atmLevel()

                for off in strike_offsets_bps:
                    strike = atm + off / 10_000
                    nvol = smile.volatility(strike, ql.Normal) * 10_000

                    col_name = f"{e}x{t} ATMF{off:+d}"
                    row[col_name] = nvol

        records.append(row)

    new_df = pd.DataFrame.from_records(records)
    new_df.insert(0, DATE_COL, pd.to_datetime(new_df.pop(DATE_COL)))
    new_df[DATE_COL] = new_df[DATE_COL].dt.tz_localize(eastern)
    new_df[DATE_COL] = new_df[DATE_COL] + pd.Timedelta(hours=17)
    new_df[DATE_COL] = new_df[DATE_COL].dt.tz_convert("UTC") 
    
    merged_df = pd.concat([curr_df, new_df])
    merged_df = merged_df.drop_duplicates(subset=[DATE_COL], keep="last")
    merged_df.set_index(DATE_COL, inplace=True)

    print(merged_df)

    merged_df.to_csv(CSV_PATH) 

    print(f"{termcolor.colored(f"Took: {time.time() - t1} sec", "blue")}")
