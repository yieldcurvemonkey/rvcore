import sys

sys.path.append("../../")

from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database, database_exists, drop_database
from termcolor import termcolor

from backend.Products.CurveBuilding.Swaptions.types import SCube, SABRParams
from backend.Products.CurveBuilding.Swaptions.swaption_data_models import Base, get_scube_cache_model, get_sabr_params_cache_model
from backend.Products.Swaps import Swaps
from backend.Products.Swaptions import Swaptions

if len(sys.argv) < 2:
    print("Usage: python update_github_vol_cube.py [init_postgres|update_postgres]")
    sys.exit(1)

mode = sys.argv[1].lower()
if mode not in ["init_postgres", "update_postgres"]:
    print("Invalid mode. Use 'init_postgres' or 'update_postgres'")
    sys.exit(1)

CURVE = "USD-SOFR-1D"

ois_db_username = "postgres"
ois_db_password = "password"
ois_db_host = "localhost"
ois_db_port = "5432"
ois_db_name = "usd_ois_cme_eris_curve"

usd_ois = Swaps(
    data_source=create_engine(f"postgresql://{ois_db_username}:{ois_db_password}@{ois_db_host}:{ois_db_port}/{ois_db_name}"),
    curve=CURVE,
    ql_interpolation_algo="log_linear",
    pre_fetch_curves=True,
    max_njobs=-1,
    error_verbose=True,
)

swaption_db_username = "postgres"
swaption_db_password = "password"
swaption_db_host = "localhost"
swaption_db_port = "5432"
swaption_db_name = "github_jpm_usd_swaptions_cube"

usd_swaptions = Swaptions(data_source="GITHUB", swaps_product=usd_ois, n_jobs=-1, error_verbose=True)

swaption_db_engine = create_engine(f"postgresql://{swaption_db_username}:{swaption_db_password}@{swaption_db_host}:{swaption_db_port}/{swaption_db_name}")

SCubeCache = get_scube_cache_model(CURVE)
SABRParamsCache = get_sabr_params_cache_model(curve=CURVE)


def update_swaption_cube_jsonb(usd_swaptions: Swaptions, start_date: datetime, end_date: datetime, engine: Engine):
    try:
        scubes_ts_dict: Dict[datetime, SCube] = usd_swaptions._fetch_scubes(start_date=start_date, end_date=end_date)
        scubes_ts_dict_pickable = {
            dt.isoformat(): {offset: grid.reset_index().to_dict("records") for offset, grid in scube.items()} for dt, scube in scubes_ts_dict.items()
        }

        inspector = inspect(engine)
        if not inspector.has_table(SCubeCache.__tablename__):
            Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        for ts, scube_pickable in scubes_ts_dict_pickable.items():
            try:
                record = SCubeCache(timestamp=ts, scube=scube_pickable)
                session.merge(record)
            except Exception as e:
                print(termcolor.colored(f"Postgres: {ts} error during write: {e}", color="red"))

        session.commit()
        session.close()

        print(
            termcolor.colored(
                f"Postgres: {start_date.date()}-{end_date.date()} SCUBE cache updated successfully in table {CURVE}",
                color="green",
            )
        )
    except Exception as e:
        print(
            termcolor.colored(
                f"Postgres: {start_date.date()}-{end_date.date()} Error updating SCUBE cache in table {CURVE}: {e}",
                color="red",
            )
        )


def update_sabr_params_jsonb(usd_swaptions: Swaptions, start_date: datetime, end_date: datetime, engine: Engine):
    try:
        sabr_params_ts_dict: Dict[datetime, SABRParams] = usd_swaptions._fetch_sabr_params(start_date=start_date, end_date=end_date)
        sabr_params_ts_dict_pickable = {dt.isoformat(): sabr_params for dt, sabr_params in sabr_params_ts_dict.items()}

        inspector = inspect(engine)
        if not inspector.has_table(SABRParamsCache.__tablename__):
            Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        for ts, sabr_params_pickable in sabr_params_ts_dict_pickable.items():
            record = SABRParamsCache(timestamp=ts, sabr_params=sabr_params_pickable)
            session.merge(record)

        session.commit()
        session.close()

        print(
            termcolor.colored(
                f"Postgres: {start_date.date()} - {end_date.date()} SABRParams cache updated successfully in table {CURVE}",
                color="green",
            )
        )
    except Exception as e:
        print(
            termcolor.colored(
                f"Postgres: {start_date.date()} - {end_date.date()} Error updating SABRParams cache in table {CURVE}: {e}",
                color="red",
            )
        )


def get_latest_scube_cache(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        latest_cache = session.query(SCubeCache).order_by(SCubeCache.timestamp.desc()).first()
        return latest_cache
    finally:
        session.close()


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


if __name__ == "__main__":

    if mode in ["init_postgres"]:
        if database_exists(swaption_db_engine.url):
            print(termcolor.colored(f"Database '{swaption_db_engine.url.database}' exists. Dropping it...", color="yellow"))
            drop_database(swaption_db_engine.url)
        if not database_exists(swaption_db_engine.url):
            print(termcolor.colored(f"Database '{swaption_db_engine.url.database}' does not exist. Creating it...", color="yellow"))
            create_database(swaption_db_engine.url)

        with swaption_db_engine.connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS hstore;"))
            connection.commit()
            print("hstore extension initialized successfully.")

        # with swaption_db_engine.connect() as connection:
        #     result = connection.execute(text("SELECT version();"))
        #     version_info = result.fetchone()[0]
        #     print("PostgreSQL version:", version_info)
        #     print("JSONB is available by default in PostgreSQL 9.4 and later.")

        start_db_date = datetime(2024, 1, 1)
        end_db_date = datetime.today()
        monthly_batches = get_date_batches(start_date=start_db_date, end_date=end_db_date, batch_days=20)
        for batch_start, batch_end in monthly_batches:
            update_swaption_cube_jsonb(usd_swaptions=usd_swaptions, start_date=batch_start, end_date=batch_end, engine=swaption_db_engine)
            update_sabr_params_jsonb(usd_swaptions=usd_swaptions, start_date=batch_start, end_date=batch_end, engine=swaption_db_engine)

    elif mode == "update_postgres":
        latest_in_store = get_latest_scube_cache(engine=swaption_db_engine).timestamp
        latest_in_store = datetime.fromisoformat(latest_in_store)
        update_swaption_cube_jsonb(usd_swaptions=usd_swaptions, start_date=latest_in_store, end_date=datetime.today(), engine=swaption_db_engine)
        update_sabr_params_jsonb(usd_swaptions=usd_swaptions, start_date=latest_in_store, end_date=datetime.today(), engine=swaption_db_engine)
