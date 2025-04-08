import sys

sys.path.append("../../")


import os
import time
from datetime import datetime, timezone
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import QuantLib as ql
import schedule
import termcolor
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy_utils import create_database, database_exists, drop_database

from core.Fetchers.DTCCFetcher import DTCCFetcher, datetime_today_utc
from core.Products.CurveBuilding.AlchemyWrapper import AlchemyWrapper

if len(sys.argv) < 2:
    print("Usage: python update_sdr.py [init_postgres|update_postgres|start_update_postgres_service]")
    sys.exit(1)

mode = sys.argv[1].lower()
if mode not in ["init_postgres", "update_postgres", "start_update_postgres_service"]:
    print("Invalid mode. Use 'init_postgres', 'update_postgres', or 'start_update_postgres_service'.")
    sys.exit(1)

dir_path = os.path.dirname(os.path.realpath(__file__))

db_username = "postgres"
db_password = "password"
db_host = "localhost"
db_port = "5432"
db_name = "sdr"

AGENCY = "CFTC"
ASSET_CLASS = "RATES"
table_name = f"{AGENCY}-{ASSET_CLASS}"

connection_string = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

UPDATE_POSTGRES_SERVICE_INTERVAL = 5

dtcc_fetcher = DTCCFetcher()
TIMESTAMP_COL = "Execution Timestamp"


def get_date_batches(start_date: datetime, end_date: datetime, batch_days: int = 90) -> List[Tuple[datetime, datetime]]:
    batches = []
    curr_start = start_date
    while curr_start < end_date:
        curr_end = curr_start + pd.Timedelta(days=batch_days)
        if curr_end > end_date:
            curr_end = end_date
        batches.append((curr_start, curr_end))
        curr_start = curr_end
    return batches


def run_update_postgres():
    try:
        alchemy_wrapper = AlchemyWrapper(engine=engine, date_col=TIMESTAMP_COL)
        curr_db_df = alchemy_wrapper.fetch_latest_row(table_name=table_name)
        curr_date_in_db: pd.Timestamp = curr_db_df.iloc[-1].name

        if curr_date_in_db.date() < datetime_today_utc().date():
            sdr_df = dtcc_fetcher.fetch_reports(
                start_date=curr_date_in_db,
                end_date=datetime.today(),
                agency=AGENCY,
                asset_class=ASSET_CLASS,
                use_pyarrow=True,
                show_tqdm=True,
            )
        else:
            sdr_df = dtcc_fetcher.fetch_intraday_reports(
                agency=AGENCY,
                asset_class=ASSET_CLASS,
                start_timestamp=curr_date_in_db.astimezone(timezone.utc),
                end_timestamp=datetime.now(timezone.utc),
                use_pyarrow=True,
                show_tqdm=True,
            )

        sdr_df = sdr_df[sdr_df[TIMESTAMP_COL] > curr_date_in_db.astimezone(timezone.utc)]
        if "report_slice" in sdr_df.columns:
            sdr_df.drop(columns=["report_slice"], inplace=True)
        sdr_df.to_sql(table_name, engine, if_exists="append", index=False)
        print(termcolor.colored(f"Postgres: Updated table '{table_name}' most recent: {sdr_df.iloc[-1][TIMESTAMP_COL]}", color="green"))

    except Exception as e:
        print(termcolor.colored(f"Postgres: Error updating table '{table_name}': {e}", color="red"))


def sdrify_df(sdr_df: pd.DataFrame):
    sdr_df["Notional amount-Leg 1"] = sdr_df["Notional amount-Leg 1"].astype(str)
    sdr_df["Notional amount-Leg 2"] = sdr_df["Notional amount-Leg 2"].astype(str)
    sdr_df["Notional amount in effect on associated effective date-Leg 1"] = sdr_df["Notional amount in effect on associated effective date-Leg 1"].astype(str)
    sdr_df["Notional amount in effect on associated effective date-Leg 2"] = sdr_df["Notional amount in effect on associated effective date-Leg 2"].astype(str)
    sdr_df["Effective date of the notional amount-Leg 1"] = sdr_df["Effective date of the notional amount-Leg 1"].astype(str)
    sdr_df["Effective date of the notional amount-Leg 2"] = sdr_df["Effective date of the notional amount-Leg 2"].astype(str)
    sdr_df["End date of the notional amount-Leg 1"] = sdr_df["End date of the notional amount-Leg 1"].astype(str)
    sdr_df["End date of the notional amount-Leg 2"] = sdr_df["End date of the notional amount-Leg 2"].astype(str)
    sdr_df["Other payment amount"] = sdr_df["Other payment amount"].astype(str)
    sdr_df["Strike Price"] = sdr_df["Strike Price"].astype(str)
    sdr_df["Spread-Leg 1"] = sdr_df["Spread-Leg 1"].astype(str)
    sdr_df["Spread-Leg 2"] = sdr_df["Spread-Leg 2"].astype(str)
    sdr_df.replace(["", " ", None, "None", "NaN"], np.nan, inplace=True)
    return sdr_df.sort_values(by="Event timestamp").reset_index(drop=True)


if __name__ == "__main__":

    if mode == "init_postgres":
        if database_exists(engine.url):
            print(termcolor.colored(f"Database '{engine.url.database}' exists. Dropping it...", color="yellow"))
            drop_database(engine.url)
        if not database_exists(engine.url):
            print(termcolor.colored(f"Database '{engine.url.database}' does not exist. Creating it...", color="yellow"))
            create_database(engine.url)

        upi_schema_migration_date = datetime(2024, 2, 5)
        today = datetime.today()
        date_batches = get_date_batches(start_date=upi_schema_migration_date, end_date=today, batch_days=5)

        for start, end in date_batches:
            historical_sdr_df = dtcc_fetcher.fetch_reports(
                start_date=start,
                end_date=end,
                agency=AGENCY,
                asset_class=ASSET_CLASS,
                use_pyarrow=True,
                show_tqdm=True,
            )
            if historical_sdr_df.empty:
                continue

            historical_sdr_df = sdrify_df(historical_sdr_df)

            try:
                historical_sdr_df.to_sql(table_name, engine, if_exists="append")
                print(termcolor.colored(f"SDR data for {start.date()} to {end.date()} successfully imported into PostgreSQL table '{table_name}'.", "green"))
            except SQLAlchemyError as e:
                print(termcolor.colored(f"Error occurred during DB init for batch {start.date()} to {end.date()}: {e}", "red"))
                print(termcolor.colored(f"Will be checking day-by-day for batch {start.date()} to {end.date()}", "yellow"))

                historical_sdr_dict_df = dtcc_fetcher.fetch_historical_reports(
                    start_date=start,
                    end_date=end,
                    agency=AGENCY,
                    asset_class=ASSET_CLASS,
                    use_pyarrow=True,
                    show_tqdm=True,
                )

                for dt, curr_sdr_df in historical_sdr_dict_df.items():
                    curr_sdr_df = sdrify_df(curr_sdr_df)

                    try:
                        curr_sdr_df.to_sql(table_name, engine, if_exists="append")
                        print(termcolor.colored(f"SDR data for {dt} successfully imported into PostgreSQL table '{table_name}'.", "green"))
                    except SQLAlchemyError as day_e:
                        print(termcolor.colored(f"Will be checking one-by-one for day {dt}: {day_e}", "yellow"))
                        max_idx = len(curr_sdr_df)
                        for idx, row in curr_sdr_df.iterrows():
                            try:
                                pd.DataFrame([row]).to_sql(table_name, engine, if_exists="append")
                                print(termcolor.colored(f"{idx}/{max_idx} for day {dt} appended", "green"))
                            except SQLAlchemyError as row_e:
                                print(termcolor.colored(f"Failed to insert row at index {idx} for day {dt}: {row_e}", "red"))

    elif mode == "update_postgres":
        run_update_postgres()

    elif mode == "start_update_postgres_service":
        print(termcolor.colored(f"Starting update_postgres service: updates every {UPDATE_POSTGRES_SERVICE_INTERVAL} minutes.", color="green"))
        run_update_postgres()
        schedule.every(UPDATE_POSTGRES_SERVICE_INTERVAL).minutes.do(run_update_postgres)
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print(termcolor.colored("Service interrupted by user. Exiting.", color="red"))
