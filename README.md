# `rvcore`: RV Analytics and Pricing Library

## Getting Started

### Init USD OIS PostgresDB

- Make sure you have Postgres on your machine

- Mind the gap

    - EOD marks after mid 2018 are CME Clearing's USD-SOFR-1D curve i.e. BBG S490
    - EOD marks before 2018 are CME Clearing's USD-OIS (FED FUNDS) i.e. BBG S42
    - Intraday marks from ERIS Discount Curve

```bash
cd .\core\DB\
```

- I'm using the default postgres user i.e. `username=postgres`, `password=password` on `port=5432`
- Change postgres user configs in script `update_cme_usd_ois_curve.py`

```py
python .\update_cme_usd_ois_curve.py init_postgres
```

### Updating the DB

```py
python .\update_cme_usd_ois_curve.py update_postgres
```

or

```py
python .\update_cme_usd_ois_curve.py start_update_postgres_service
```

- runs `update_postgres` every minute for intraday monitoring

## Roadmap