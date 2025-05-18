from datetime import datetime
from typing import Dict, List, Literal, Optional

import pandas as pd
from sqlalchemy import Engine, text


class AlchemyWrapper:
    _engine: Engine = None
    _date_col: str = None
    _index_col: str = None

    def __init__(
        self,
        engine: Engine,
        date_col: Optional[str] = "Date",
        index_col: Optional[str] = None,
    ):
        self._engine = engine
        self._date_col = date_col
        self._index_col = index_col
        if not self._index_col:
            self._index_col = self._date_col


    def fetch_latest_row(self, table_name: str, cols: Optional[List[str]] = None, set_index: Optional[bool] = True, utc: Optional[bool] = False) -> pd.DataFrame:
        selected_columns = ", ".join(cols) if cols else "*"
        table_identifier = f'"{table_name}"'
        date_column = f'"{self._date_col}"'

        query = text(f"SELECT {selected_columns} FROM {table_identifier} ORDER BY {date_column} DESC LIMIT 1")
        df = pd.read_sql_query(query, self._engine)

        if self._date_col in df.columns:
            df[self._date_col] = pd.to_datetime(df[self._date_col], errors="coerce", utc=utc)
        if set_index:
            df.set_index(self._index_col, inplace=True)

        return df

    def _fetch_df_by_dates(
        self,
        table_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        cols: Optional[List[str]] = None,
        set_index: Optional[bool] = True,
        utc: Optional[bool] = False,
    ) -> pd.DataFrame:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"

        selected_columns = ", ".join(cols) if cols else "*"
        table_identifier = f'"{table_name}"'
        date_column = f'"{self._date_col}"'

        if start_date and end_date:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            query = text(f"SELECT {selected_columns} FROM {table_identifier} WHERE CAST({date_column} AS DATE) BETWEEN :start_date AND :end_date")
            params = {"start_date": start_str, "end_date": end_str}
        else:
            placeholders = []
            params = {}
            for i, d in enumerate(bdates):
                key = f"date_{i}"
                placeholders.append(f":{key}")
                params[key] = d.strftime("%Y-%m-%d")
            in_clause = ", ".join(placeholders)
            query = text(f"SELECT {selected_columns} FROM {table_identifier} " f"WHERE CAST({date_column} AS DATE) IN ({in_clause})")

        df = pd.read_sql_query(query, self._engine, params=params)
        if self._date_col in df.columns:
            df[self._date_col] = pd.to_datetime(df[self._date_col], errors="coerce", utc=utc)
        if set_index:
            df.set_index(self._index_col, inplace=True)

        return df

    def _fetch_by_col_values(
        self,
        table_name: str,
        search_params_dict: Dict[str, List[str]],
        set_index: Optional[bool] = True,
        and_or: Literal["and", "or"] = "and",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        cols_to_look: Optional[List[str]] = None,
        utc: Optional[bool] = False,
    ) -> pd.DataFrame:
        if len(search_params_dict) == 0 and start_date and end_date and cols_to_look is None:
            return self._fetch_df_by_dates(table_name=table_name, start_date=start_date, end_date=end_date, set_index=set_index, utc=utc)

        if cols_to_look is not None:
            cols_to_look = list(set(cols_to_look))
            selected_columns = ", ".join([f'"{col}"' for col in cols_to_look])
        else:
            selected_columns = "*"

        table_identifier = f'"{table_name}"'

        conditions = []
        params = {}
        for col, values in search_params_dict.items():
            if values:
                placeholders = []
                for i, val in enumerate(values):
                    key = f"{col}_val_{i}"
                    placeholders.append(f":{key}")
                    params[key] = val
                col_identifier = f'"{col}"'
                conditions.append(f"{col_identifier} IN ({', '.join(placeholders)})")

        joiner = " AND " if and_or.lower() == "and" else " OR "
        if conditions:
            final_where_clause = " WHERE " + joiner.join(f"({cond})" for cond in conditions)
        else:
            final_where_clause = ""

        query = text(f"SELECT {selected_columns} FROM {table_identifier}{final_where_clause}")
        df = pd.read_sql_query(query, self._engine, params=params)

        if self._date_col in df.columns:
            df[self._date_col] = pd.to_datetime(df[self._date_col], errors="coerce", utc=utc)
            if start_date:
                df = df[df[self._date_col].dt.date >= start_date.date()]
            if end_date:
                df = df[df[self._date_col].dt.date <= end_date.date()]
            if bdates:
                df = df[df[self._date_col].dt.date.isin([dt.date() for dt in bdates])]

        if set_index:
            df.set_index(self._index_col, inplace=True)

        return df
