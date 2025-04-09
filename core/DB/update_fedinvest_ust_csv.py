import sys

sys.path.append("../../")

import os
import time
from datetime import datetime
from typing import List, Set, Tuple

import pandas as pd
import QuantLib as ql
import schedule
import termcolor
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy_utils import create_database, database_exists, drop_database

from core.DB.FedInvestDataBuilder import FedInvestDataBuilder
from core.Products.CurveBuilding.AlchemyWrapper import AlchemyWrapper
from core.utils.ql_utils import datetime_to_ql_date, most_recent_business_day_from_date, ql_date_to_datetime

if len(sys.argv) < 2:
    print("Usage: python update_fedinvest_ust_csv.py [init_postgres|update_postgres|update_csv|start_update_postgres_service]")
    sys.exit(1)

mode = sys.argv[1].lower()
if mode not in ["init_postgres", "update_postgres", "update_csv", "start_update_postgres_service"]:
    print("Invalid mode. Use 'init_postgres', 'update_postgres', 'update_csv', or 'start_update_postgres_service'.")
    sys.exit(1)

dir_path = os.path.dirname(os.path.realpath(__file__))
FEDINVEST_CUSIP_CSV = rf"dump/fedinvest_cusip_set.csv"
CT_TIMESERIES_CSV = rf"dump/ct_timeseries.csv"

db_username = "postgres"
db_password = "password"
db_host = "localhost"
db_port = "5432"
db_name = "rvcore_fedinvest_cusip_set"
table_name = "USD"
connection_string = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

UPDATE_POSTGRES_SERVICE_INTERVAL = 5

ql_calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)


def get_date_batches(start_date: datetime, end_date: datetime, batch_days: int = 90) -> List[Tuple[datetime, datetime]]:
    batches = []
    curr_start = start_date.date()
    clean_end_date = end_date.date()

    while curr_start < clean_end_date:
        curr_end = curr_start + pd.Timedelta(days=batch_days)
        if curr_end > clean_end_date:
            curr_end = clean_end_date
        batches.append((curr_start, curr_end))
        curr_start = curr_end
    return batches


def deduplicate_by_day(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed", utc=True)
    # df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    # df["timestamp"] = df["timestamp"].apply(lambda x: x.tz_localize(None) if x is not None and getattr(x, "tzinfo", None) is not None else x)
    df.sort_values("timestamp", inplace=True)
    grouped_dfs = df.groupby("timestamp")

    eod_dfs = {}
    intraday_dfs = {}
    for ts, group in grouped_dfs:
        if ts.time() == datetime.min.time() or ts.time() == None:
            eod_dfs[ts.date()] = group
        else:
            intraday_dfs[ts.date()] = group

    dedup_dts: Set[datetime] = set(list(eod_dfs.keys()) + list(intraday_dfs.keys()))
    dedup_dfs = []
    for d in dedup_dts:
        if d in eod_dfs and d in intraday_dfs:
            dedup_dfs.append(eod_dfs[d])
        elif d in eod_dfs and not d in intraday_dfs:
            dedup_dfs.append(eod_dfs[d])
        elif not d in eod_dfs and d in intraday_dfs:
            dedup_dfs.append(intraday_dfs[d])
        else:
            pass

    return pd.concat(dedup_dfs, ignore_index=True)


def update_csv_and_get_df() -> pd.DataFrame:
    try:
        if os.path.exists(FEDINVEST_CUSIP_CSV) and os.path.getsize(FEDINVEST_CUSIP_CSV) > 0:
            curr_df = pd.read_csv(FEDINVEST_CUSIP_CSV)
            curr_df["timestamp"] = pd.to_datetime(curr_df["timestamp"], errors="coerce", format="mixed")
            curr_df.sort_values("timestamp", inplace=True)
            last_date = curr_df.iloc[-1]["timestamp"]
        else:
            last_date = datetime(2020, 1, 1)
            curr_df = pd.DataFrame()

        today = datetime.now()
        fedinvest_data_builder = FedInvestDataBuilder()

        batches = get_date_batches(last_date, today, batch_days=20)
        all_data_frames = []
        for batch_start, batch_end in batches:
            if not ql_calendar.isBusinessDay(ql.Date(batch_start.day, batch_start.month, batch_start.year)):
                batch_start = most_recent_business_day_from_date(batch_start, ql_calendar, tz="America/New_York", to_pydate=True)
            if not ql_calendar.isBusinessDay(ql.Date(batch_end.day, batch_end.month, batch_end.year)):
                batch_end = most_recent_business_day_from_date(batch_end, ql_calendar, tz="America/New_York", to_pydate=True)

            print(termcolor.colored(f"Fetching data from {batch_start.date()} to {batch_end.date()}", color="blue"))
            new_data_dict = fedinvest_data_builder.fetch_historical_curve_sets(
                start_date=batch_start,
                end_date=batch_end,
                fetch_soma_holdings=True,
                fetch_stripping_data=True,
                calc_free_float=True,
                pricer_n_jobs=-1,
                sorted_curve_set=True,
            )
            if new_data_dict:
                batch_df = pd.concat(new_data_dict.values(), ignore_index=True)
                batch_df["timestamp"] = batch_df["timestamp"].apply(lambda x: x.tz_localize(None) if x is not None and getattr(x, "tzinfo", None) is not None else x)
                all_data_frames.append(batch_df)

        new_data_df = pd.concat(all_data_frames, ignore_index=True) if all_data_frames else pd.DataFrame()
        combined_df = pd.concat([curr_df, new_data_df], ignore_index=True) if not curr_df.empty else new_data_df
        deduped_df = deduplicate_by_day(combined_df)
        if "level_0" in deduped_df.columns:
            deduped_df.drop(columns=["level_0"], inplace=True)

        deduped_df.reset_index(drop=True, inplace=True)
        deduped_df.drop_duplicates(subset=["hash"], keep="last", inplace=True)
        deduped_df.sort_values(by=["timestamp"], inplace=True)

        deduped_df.to_csv(FEDINVEST_CUSIP_CSV, index=False)
        print(termcolor.colored(f"Cash Curve: SUCCESSFULLY WROTE to {FEDINVEST_CUSIP_CSV}", color="green"))
        return deduped_df

    except Exception as e:
        print(termcolor.colored(f"Cash Curve: SOMETHING WENT WRONG: {e}", color="red"))
        sys.exit(1)


def run_update_postgres():
    try:
        query = text(f'SELECT * FROM "{table_name}" ORDER BY "timestamp" DESC LIMIT 1')
        with engine.connect() as conn:
            last_row = pd.read_sql(query, conn)

        if not last_row.empty:
            last_row["timestamp"] = pd.to_datetime(last_row["timestamp"], errors="coerce", format="mixed")
            last_date = last_row.iloc[0]["timestamp"]

            achemy_wrapper = AlchemyWrapper(engine=engine, date_col="timestamp", index_col="hash")
            result_df = achemy_wrapper._fetch_df_by_dates(table_name=table_name, start_date=last_date, end_date=last_date, set_index=False)
            last_date_to_fetch = ql_date_to_datetime(ql_calendar.advance(datetime_to_ql_date(last_date), ql.Period("-3D")))
        else:
            last_date_to_fetch = datetime(2020, 1, 1)

        fedinvest_data_builder = FedInvestDataBuilder()
        today = datetime.today()

        batches = get_date_batches(last_date_to_fetch, today, batch_days=90)
        all_data_frames = []
        for batch_start, batch_end in batches:
            if not ql_calendar.isBusinessDay(ql.Date(batch_start.day, batch_start.month, batch_start.year)):
                batch_start = most_recent_business_day_from_date(batch_start, ql_calendar, tz="US/Eastern", to_pydate=True)
            if not ql_calendar.isBusinessDay(ql.Date(batch_end.day, batch_end.month, batch_end.year)):
                batch_end = most_recent_business_day_from_date(batch_end, ql_calendar, tz="US/Eastern", to_pydate=True)

            print(termcolor.colored(f"Fetching new data from {batch_start.date()} to {batch_end.date()}", color="blue"))
            new_data_dict = fedinvest_data_builder.fetch_historical_curve_sets(
                start_date=batch_start,
                end_date=batch_end,
                fetch_soma_holdings=True,
                fetch_stripping_data=True,
                calc_free_float=True,
                pricer_n_jobs=-1,
                sorted_curve_set=True,
            )
            if new_data_dict:
                batch_df = pd.concat(new_data_dict.values(), ignore_index=True)
                batch_df["timestamp"] = batch_df["timestamp"].apply(lambda x: x.tz_localize(None) if x is not None and getattr(x, "tzinfo", None) is not None else x)
                all_data_frames.append(batch_df)

        new_data_df = pd.concat(all_data_frames, ignore_index=True) if all_data_frames else pd.DataFrame()
        combined_df = pd.concat([result_df, new_data_df], ignore_index=True) if not result_df.empty else new_data_df
        combined_df.drop_duplicates(subset=["hash"])

        deduped_df = deduplicate_by_day(combined_df)

        if "level_0" in deduped_df.columns:
            deduped_df.drop(columns=["level_0"], inplace=True)

        deduped_df.reset_index(drop=True, inplace=True)
        deduped_df.drop_duplicates(subset=["hash"], keep="last", inplace=True)
        deduped_df.sort_values(by=["timestamp"], inplace=True)

        delete_date = deduped_df["timestamp"].min()
        with engine.begin() as conn:
            delete_query = text(f'DELETE FROM "{table_name}" WHERE "timestamp" >= \'{pd.to_datetime(delete_date).strftime("%Y-%m-%d")}\'')
            conn.execute(delete_query)

        deduped_df.to_sql(table_name, engine, if_exists="append", index=False)
        print(termcolor.colored(f"Postgres: Data updated successfully in table '{table_name}'. Most recent record: {deduped_df['timestamp'].max()}", color="green"))

    except Exception as e:
        print(termcolor.colored(f"Postgres: Error updating table '{table_name}': {e}", color="red"))

    try:
        fedinvest_data_builder = FedInvestDataBuilder()
        ct_ts_df = fedinvest_data_builder.fetch_otr_timeseries()
        ct_ts_df.index.names = ["timestamp"]
        ct_ts_df.to_sql(f"{table_name}_otr_timeseries", engine, if_exists="replace")
        print(
            termcolor.colored(f"Postgres: Data updated successfully in table '{table_name}_otr_timeseries'. Most recent record: {ct_ts_df.index[-1]}", color="green")
        )
    except Exception as e:
        print(termcolor.colored(f"Postgres: Error updating table '{table_name}_otr_timeseries': {e}", color="red"))


def update_ct_timeseries_csv():
    try:
        fedinvest_data_builder = FedInvestDataBuilder()
        ct_ts_df = fedinvest_data_builder.fetch_otr_timeseries()
        ct_ts_df.index.names = ["timestamp"]
        ct_ts_df.to_csv(CT_TIMESERIES_CSV)
        print(termcolor.colored(f"CT Timeseries: SUCCESSFULLY WROTE to {CT_TIMESERIES_CSV}", color="green"))
        return ct_ts_df
    except Exception as e:
        print(termcolor.colored(f"CT Timeseries: SOMETHING WENT WRONG: {e}", color="red"))
        sys.exit(1)


if __name__ == "__main__":

    if mode in ["update_csv", "init_postgres"]:
        updated_df = update_csv_and_get_df()
        ct_ts_df = update_ct_timeseries_csv()

        if mode == "update_csv":
            sys.exit(0)
        else:
            try:
                if database_exists(engine.url):
                    print(termcolor.colored(f"Database '{engine.url.database}' exists. Dropping it...", color="yellow"))
                    drop_database(engine.url)
                if not database_exists(engine.url):
                    print(termcolor.colored(f"Database '{engine.url.database}' does not exist. Creating it...", color="yellow"))
                    create_database(engine.url)

                updated_df.to_sql("USD", engine, if_exists="replace", index=False)
                print(f"CSV data successfully imported into PostgreSQL table '{table_name}'.")

                ct_ts_df.to_sql("USD_otr_timeseries", engine, if_exists="replace", index=False)
                print(f"CSV data successfully imported into PostgreSQL table '{table_name}'.")

            except SQLAlchemyError as e:
                print(f"Error occurred during DB init: {e}")

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
