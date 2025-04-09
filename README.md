# `rvcore`: Rates RV Analytics and Pricing Python SDK

## Getting Started

- Make sure you have Postgres on your machine
- I'm using the default postgres user i.e. `username=postgres`, `password=password` on `port=5432`
- TODO: `.env` file for all the postgres config stuff

- Install SDK dependancies

```bash
pip install -r requirements.txt
```

---

### Init USD OIS PostgresDB


- Mind the gap:
    - EOD marks after mid 2018 are CME Clearing's USD-SOFR-1D curve i.e. BBG S490
    - EOD marks before 2018 are CME Clearing's USD-OIS (FED FUNDS) i.e. BBG S42
    - Intraday marks from the ERIS Discount Curve

```bash
cd .\backend\DB\
```

- **Script will drop `db name=rvcore_usd_ois_cme_eris_curve` if it already exists**

```py
python .\update_cme_usd_ois_curve.py init_postgres
```


#### Updating the DB

```py
python .\update_cme_usd_ois_curve.py update_postgres
```

#### Starting intraday microservice

```py
python .\update_cme_usd_ois_curve.py start_update_postgres_service
```

- runs `update_postgres` every minute for intraday monitoring

---

### Init USD Cash (USTs) PostgresDB

- FedInvest EOD Marks (NY Close)
- Intraday data scraped from WSJ (ICAP)

```bash
cd .\backend\DB\
```

- **Script will drop `db name=rvcore_fedinvest_cusip_set` if it already exists**

```py
python .\update_fedinvest_ust_csv.py init_postgres
```

#### Updating the DB

```py
python .\update_fedinvest_ust_csv.py update_postgres
```

#### Starting intraday microservice

```py
python .\update_fedinvest_ust_csv.py start_update_postgres_service
```

- runs `update_postgres` every minute for intraday monitoring


---

### Init USD Swaptions PostgresDB

- Built from JPM Analytics Packages
- SOFR OIS (BBG ICVS 490 curve) NY Close
- SABR Model calibrated using [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) with a Residual Sum of Squares objective function and a refinement using simple Gauss-Newton to minimize the weighted root mean square error in implied volatilities as described in `Le Floc’h` + `Kennedy`'s [Explicit SABR Calibration through Simple Expansions](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2467231)

- Data source: https://github.com/yieldcurvemonkey/MONKEY_CUBE

```bash
cd .\backend\DB\
```

- **Script will drop `db name=rvcore_github_jpm_usd_swaptions_cube` if it already exists**

```py
python .\update_github_vol_cube.py init_postgres
```

#### Updating the DB

```py
python .\update_github_vol_cube.py update_postgres
```

---

### SDR PostgresDB

- DTCC SDR CFTC-Rates DB
- Data only avaliable after UPI migration

```bash
cd .\backend\DB\
```

- - **Script will drop `db name=rvcore_sdr` if it already exists**

```py
python .\update_sdr.py init_postgres
```

#### Updating the DB

```py
python .\update_sdr.py update_postgres
```

#### Starting intraday microservice

```py
python .\update_sdr.py start_update_postgres_service
```


## Roadmap