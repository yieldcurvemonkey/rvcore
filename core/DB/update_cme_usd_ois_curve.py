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

from core.Products.CurveBuilding.AlchemyWrapper import AlchemyWrapper
from core.Products.CurveBuilding.ql_curve_building_utils import get_nodes_dict
from core.Products.CurveBuilding.Swaps.swap_data_models import Base, get_ql_curve_cache_model
from core.Products.Swaps import Swaps

if len(sys.argv) < 2:
    print("Usage: python update_cme_usd_ois_curve.py [init_postgres|update_postgres|update_csv|start_update_postgres_service]")
    sys.exit(1)

mode = sys.argv[1].lower()
if mode not in ["init_postgres", "update_postgres", "update_csv", "start_update_postgres_service"]:
    print("Invalid mode. Use 'init_postgres', 'update_postgres', 'update_csv', or 'start_update_postgres_service'.")
    sys.exit(1)

dir_path = os.path.dirname(os.path.realpath(__file__))
USD_OIS_CSV_PATH = rf"{dir_path}/dump/hist_usd_ois_cme_eris_curve.csv"

db_username = "postgres"
db_password = "password"
db_host = "localhost"
db_port = "5432"
db_name = "rvcore_usd_ois_cme_eris_curve"
connection_string = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

UPDATE_POSTGRES_SERVICE_INTERVAL = 1

chi_now = datetime.now(tz=pytz.timezone("US/Central"))
chi_today = datetime(year=chi_now.year, month=chi_now.month, day=chi_now.day)

USD_OIS_CURVE = "USD-SOFR-1D"
INTERPOLATION_ALGO = "log_linear"

usd_swaps = Swaps(
    data_source="CME",
    curve=USD_OIS_CURVE,
    ql_interpolation_algo=INTERPOLATION_ALGO,
    show_tqdm=True,
    error_verbose=True,
    max_njobs=-1,
)

par_tenors_to_fetch = [
    (f"SWAP_0Dx{t}", t)
    for t in [
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
        "13M",
        "14M",
        "15M",
        "16M",
        "17M",
        "18M",
        "19M",
        "20M",
        "21M",
        "22M",
        "23M",
        "2Y",
        "3Y",
        "4Y",
        "5Y",
        "6Y",
        "7Y",
        "8Y",
        "9Y",
        "10Y",
        "15Y",
        "20Y",
        "25Y",
        "30Y",
        "40Y",
        "50Y",
    ]
]


def deduplicate_by_day(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    df = df.sort_values("Date")

    grouped = df.groupby(df["Date"].dt.date)
    results = []
    for _, group in grouped:
        midnight_mask = group["Date"].dt.time == datetime.min.time()
        if midnight_mask.any():
            chosen = group.loc[midnight_mask].iloc[-1]
        else:
            chosen = group.iloc[-1]
        results.append(chosen)
    deduped = pd.DataFrame(results)
    deduped = deduped.set_index("Date")
    return deduped


def update_csv_and_get_df():
    try:
        curr_usd_sofr_1d_curve_df = pd.read_csv(USD_OIS_CSV_PATH)
        curr_usd_sofr_1d_curve_df["Date"] = pd.to_datetime(curr_usd_sofr_1d_curve_df["Date"], errors="coerce", format="mixed")
        curr_usd_sofr_1d_curve_df.set_index("Date", inplace=True)
        curr_usd_sofr_1d_curve_df.sort_index(inplace=True)
        curr_date_in_csv = curr_usd_sofr_1d_curve_df.iloc[-1].name

        swap_ts_df = usd_swaps.swaps_timeseries_builder(
            start_date=datetime(curr_date_in_csv.year, curr_date_in_csv.month, curr_date_in_csv.day),
            end_date=chi_today,
            cols=par_tenors_to_fetch,
            n_jobs=1,
        )

        if isinstance(swap_ts_df.index, pd.DatetimeIndex) and swap_ts_df.index.tz is not None:
            swap_ts_df.index = swap_ts_df.index.tz_localize(None)

        combined_df = pd.concat([curr_usd_sofr_1d_curve_df, swap_ts_df])
        combined_df = combined_df.reset_index()
        combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce", format="mixed")
        deduped_df = deduplicate_by_day(combined_df)
        deduped_df.to_csv(USD_OIS_CSV_PATH)
        print(termcolor.colored(f"USD SWAPS: SUCCESSFULLY WROTE to {USD_OIS_CSV_PATH}", color="green"))
        return curr_usd_sofr_1d_curve_df

    except Exception as e:
        print(termcolor.colored(f"USD SWAPS: SOMETHING WENT WRONG: {e}", color="red"))
        sys.exit(1)


def update_discount_curve_hstore(usd_swaps: Swaps, chi_today: datetime, engine: Engine):
    try:
        ql_curve_cache_pickable = {
            ts.isoformat(): get_nodes_dict(ql_curve) for ts, ql_curve in usd_swaps._ql_curve_cache.items() if ts.date() != chi_today.date()
        }  # dont cache intraday snap

        def convert_dict_for_hstore(ql_curve_cache_pickable):
            converted = {}
            for ts, nodes in ql_curve_cache_pickable.items():
                converted_nodes = {node.isoformat(): str(val) for node, val in nodes.items()}
                converted[ts] = converted_nodes
            return converted

        hstore_ready_data = convert_dict_for_hstore(ql_curve_cache_pickable)

        QLCurveCache = get_ql_curve_cache_model(curve=usd_swaps._curve, interpolation=usd_swaps._ql_interpolation_algo)

        inspector = inspect(engine)
        if not inspector.has_table(QLCurveCache.__tablename__):
            Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        for ts, nodes in hstore_ready_data.items():
            record = QLCurveCache(timestamp=ts, nodes=nodes)
            session.merge(record)  # merge handles insert or update
        session.commit()
        session.close()

        print(
            termcolor.colored(
                f"Postgres: Discount curve cache table updated successfully in table '{QLCurveCache.__tablename__}'",
                color="green",
            )
        )
    except Exception as e:
        print(termcolor.colored(f"Postgres: Error updating discount curve cache table '{QLCurveCache.__tablename__}': {e}", color="red"))


def run_update_postgres():
    try:
        alchemy_swaps_wrapper = AlchemyWrapper(engine=engine)
        curr_db_df = alchemy_swaps_wrapper.fetch_latest_row(table_name=USD_OIS_CURVE)
        curr_date_in_db = curr_db_df.iloc[-1].name

        swap_ts_df = usd_swaps.swaps_timeseries_builder(
            start_date=datetime(curr_date_in_db.year, curr_date_in_db.month, curr_date_in_db.day),
            end_date=chi_today,
            cols=par_tenors_to_fetch,
            n_jobs=1,
        )

        combined_df = pd.concat([curr_db_df, swap_ts_df])
        combined_df = combined_df.reset_index()
        combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce", format="mixed", utc=True)
        deduped_df = deduplicate_by_day(combined_df)

        with engine.begin() as conn:
            delete_query = text(f"DELETE FROM \"{USD_OIS_CURVE}\" WHERE \"Date\" >= '{curr_date_in_db.strftime('%Y-%m-%d')}'")
            conn.execute(delete_query)
        deduped_df.to_sql(USD_OIS_CURVE, engine, if_exists="append", index=True)
        print(termcolor.colored(f"Postgres: Data updated successfully in table '{USD_OIS_CURVE}'. Most recent record: {deduped_df.iloc[-1].name}", color="green"))
    except Exception as e:
        print(termcolor.colored(f"Postgres: Error updating table '{USD_OIS_CURVE}': {e}", color="red"))

    update_discount_curve_hstore(usd_swaps=usd_swaps, chi_today=chi_today, engine=engine)


if __name__ == "__main__":

    if mode in ["update_csv", "init_postgres"]:
        updated_df = update_csv_and_get_df()

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

                updated_df.to_sql(USD_OIS_CURVE, engine, if_exists="replace", index=True)
                print(f"CSV data successfully imported into PostgreSQL table '{USD_OIS_CURVE}'.")
            except SQLAlchemyError as e:
                print(f"Error occurred during DB init: {e}")

            new_usd_swaps = Swaps(
                data_source=engine,
                curve=USD_OIS_CURVE,
                ql_interpolation_algo="log_linear",
                show_tqdm=True,
                error_verbose=True,
                max_njobs=-1,
            )
            new_usd_swaps.swaps_timeseries_builder(
                start_date=datetime(2020, 1, 1),
                end_date=chi_today,
                cols=par_tenors_to_fetch,
                n_jobs=-1,
            )

            with engine.connect() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS hstore;"))
                connection.commit()
                print("hstore extension initialized successfully.")

            update_discount_curve_hstore(usd_swaps=new_usd_swaps, chi_today=chi_today, engine=engine)

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
