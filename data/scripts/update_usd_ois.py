import sys

sys.path.append("../../")

import os
import time
from datetime import datetime

import pandas as pd
import pytz
import schedule
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database, database_exists, drop_database
from termcolor import termcolor

from core.IRSwaps import IRSwaps, IRSwapQuery, IRSwapValue, IRSwapStructure, IRSwapStructureFunctionMap, IRSwapValueFunctionMap
from core.IRSwaptions import IRSwaptions, IRSwaptionQuery, IRSwaptionValue, IRSwaptionStructure, IRSwaptionStructureFunctionMap, IRSwaptionValueFunctionMap
from core.DataFetching.FixingsFetcher import FixingsFetcher
from core.utils.ql_utils import ql_date_to_datetime, datetime_to_ql_date

valid_clas = ["update_csv"]

if len(sys.argv) < 2:
    print(f"Usage: python update_usd_ois.py {valid_clas}")
    sys.exit(1)

mode = sys.argv[1].lower()
if mode not in valid_clas:
    print(f"Invalid mode. Use one of {valid_clas}")
    sys.exit(1)


chi_now = datetime.now(tz=pytz.timezone("US/Central"))
chi_today = datetime(year=chi_now.year, month=chi_now.month, day=chi_now.day)

CURVE = "USD-SOFR-1D"
INTERPOLATION = "log_linear"
USD_OIS_CSV_PATH = f"{os.path.join(os.getcwd(), "..", "usd_ois.csv")}"
DATE_COL = "Date" 
PAR_TENORS = [
    "1D",
    "1W",
    "2W",
    "3W",
    "1M",
    "2M",
    "3M",
    "4M",
    "5M",
    "6M",
    "7M",
    "8M",
    "9M",
    "10M",
    "11M",
    "12M",
    "15M",
    "18M",
    "21M",
    "2Y",
    "3Y",
    "4Y",
    "5Y",
    "6Y",
    "7Y",
    "8Y",
    "9Y",
    "10Y",
    "12Y",
    "15Y",
    "20Y",
    "25Y",
    "30Y",
    "40Y",
    "50Y",
]


def _deduplicate_by_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    def pick_one(group: pd.DataFrame) -> pd.Series:
        not_17 = group[group[DATE_COL].dt.hour != 17]
        if not not_17.empty:
            idx = not_17[DATE_COL].idxmax()
        else:
            idx = group[DATE_COL].idxmax()
        return group.loc[idx]

    deduped = df.groupby(df[DATE_COL].dt.date, group_keys=False).apply(pick_one).reset_index(drop=True).sort_values(DATE_COL)
    return deduped


def update_csv():

    usd_ois_cme = IRSwaps(
        curve=CURVE,
        data_source="CME",
        ql_interpolation_algo="log_linear",
        error_verbose=True,
        max_njobs=-1,
    )

    try:
        curr_usd_ois_df = pd.read_csv(USD_OIS_CSV_PATH)
        curr_usd_ois_df[DATE_COL] = pd.to_datetime(curr_usd_ois_df[DATE_COL])
        curr_usd_ois_df.set_index(DATE_COL, inplace=True)
        curr_usd_ois_df.sort_index(inplace=True)
        curr_date_in_csv = curr_usd_ois_df.iloc[-1].name

        ts_df = usd_ois_cme.irswaps_timeseries_builder(
            start_date=datetime(curr_date_in_csv.year, curr_date_in_csv.month, curr_date_in_csv.day),
            end_date=chi_today,
            queries=[(IRSwapQuery(tenor=t), t) for t in PAR_TENORS],
            n_jobs=-1,
        )

        curr_usd_ois_df.index = curr_usd_ois_df.index.map(lambda ts: ts.isoformat())
        ts_df.index = ts_df.index.map(lambda ts: ts.isoformat())

        combined_df = pd.concat([curr_usd_ois_df, ts_df])
        combined_df = combined_df.reset_index()
        combined_df[DATE_COL] = pd.to_datetime(combined_df[DATE_COL], utc=True)
        deduped_df = _deduplicate_by_day(combined_df)
        deduped_df = deduped_df.set_index(DATE_COL)

        deduped_df.to_csv(USD_OIS_CSV_PATH)
        print(termcolor.colored(f"USD SWAPS: SUCCESSFULLY WROTE to {USD_OIS_CSV_PATH}", color="green"))
        return deduped_df

    except Exception as e:
        print(termcolor.colored(f"USD SWAPS: SOMETHING WENT WRONG: {e}", color="red"))
        sys.exit(1)


if __name__ == "__main__":

    if mode in ["update_csv"]:
        update_csv()
        sys.exit(0)
