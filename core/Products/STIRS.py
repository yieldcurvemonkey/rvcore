from datetime import datetime
from typing import Dict, List, Optional, Literal, Tuple, Union

import re
import numpy as np
import pandas as pd
import QuantLib as ql

from Products.BaseProductPlotter import BaseProductPlotter

from core.Fetchers.BarchartFetcher import BarchartFetcher, cme_curve_to_barchart_stirs_futures_symbol

from utils.ql_utils import ql_period_to_months


class STIRS(BaseProductPlotter):
    _futures_data_source: Literal["BAR", "JPM", "SKY"] = None
    _curve: str = None
    _ql_interpolation_algo: str = None

    _futures_data_fetcher: BarchartFetcher = None

    _show_tqdm: bool = None
    _proxies: Dict[str, str] = None

    _DEFAULT_SWAP_PREFIX = "STIRS"
    _FETCH_FUNC_SYMBOLS = "_STIRS_SYMBOLS"
    _FETCH_FUNC_START_DATE = "_START_DATE"
    _FETCH_FUNC_END_DATE = "_END_DATE"

    _FETCHER_TIMEOUT = 10
    _MAX_CONNECTIONS = 64
    _MAX_KEEPALIVE_CONNECTIONS = 5

    def __init__(
        self,
        futures_data_source: Literal["BAR", "JPM", "SKY"],
        curve: Literal[
            "USD-SOFR-1D",
            "USD-FEDFUNDS",
            "JPY-TONAR",
            "CAD-CORRA",
            "EUR-ESTR",
            "EUR-EURIBOR-1M",
            "EUR-EURIBOR-3M",
            "EUR-EURIBOR-6M",
            "GBP-SONIA",
            "CHF-SARON-1D",
            "NOK-NIBOR-6M",
            "HKD-HIBOR-3M",
            "AUD-AONIA",
            "SGD-SORA-1D",
        ],
        show_tqdm: Optional[bool] = True,
        proxies: Optional[Dict[str, str]] = None,
        info_verbose: Optional[bool] = False,
        debug_verbose: Optional[bool] = False,
        warning_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        super().__init__(
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            warning_verbose=warning_verbose,
            error_verbose=error_verbose,
        )

        if futures_data_source in ["SKY", "SDR"]:
            raise NotImplementedError(f"data source not implemented")

        self._futures_data_source = futures_data_source
        self._curve = curve

        self._show_tqdm = show_tqdm
        self._proxies = proxies

        self._info_verbose = info_verbose
        self._debug_verbose = debug_verbose
        self._warning_verbose = warning_verbose
        self._error_verbose = error_verbose

        self.__data_sources_funcs = {
            "HIST_STIRS": {
                "BAR": {
                    "obj": BarchartFetcher,
                    "init_args": (
                        self._FETCHER_TIMEOUT,
                        self._proxies,
                        self._debug_verbose,
                        self._info_verbose,
                        self._warning_verbose,
                        self._error_verbose,
                    ),
                    "fetch_func": "barchart_timeseries_api",
                    "fetch_func_args": (
                        self._FETCH_FUNC_SYMBOLS,
                        self._FETCH_FUNC_START_DATE,
                        self._FETCH_FUNC_END_DATE,
                        None,
                        self._MAX_CONNECTIONS,
                        True,
                        self._show_tqdm,
                        "Close",
                    ),
                    "symbol_translator_func": cme_curve_to_barchart_stirs_futures_symbol,
                },
            },
        }
        self._futures_data_fetcher = self.__data_sources_funcs["HIST_STIRS"][self._futures_data_source]["obj"](
            *self.__data_sources_funcs["HIST_STIRS"][self._futures_data_source]["init_args"]
        )

    def _timeseries_builder_helper(self, start_date: datetime, end_date: datetime, month_codes: List[str], implied_rate: Optional[bool] = False):
        formated_symbols = [
            f"{self.__data_sources_funcs["HIST_STIRS"][self._futures_data_source]["symbol_translator_func"](self._curve)}{month_code}" for month_code in month_codes
        ]
        hist_stirs_fetch_func: callable = getattr(self._futures_data_fetcher, self.__data_sources_funcs["HIST_STIRS"][self._futures_data_source]["fetch_func"])
        hist_stirs_fetch_func_args = tuple(
            (
                start_date
                if arg == self._FETCH_FUNC_START_DATE
                else end_date if arg == self._FETCH_FUNC_END_DATE else formated_symbols if arg == self._FETCH_FUNC_SYMBOLS else arg
            )
            for arg in self.__data_sources_funcs["HIST_STIRS"][self._futures_data_source]["fetch_func_args"]
        )
        ts_df = hist_stirs_fetch_func(*hist_stirs_fetch_func_args)
        ts_df.columns = month_codes[: len(ts_df.columns)]
        if implied_rate:
            for col in ts_df.columns:
                ts_df[col] = 1 - ts_df[col]
        return ts_df

    def timeseries_builder(self, start_date: datetime, end_date: datetime, cols: List[str | Tuple[str, str]], implied_rate: Optional[bool] = False) -> pd.DataFrame:
        month_codes = set()
        for col in cols:
            if isinstance(col, tuple):
                col = col[0]
            month_codes.update(self._extract_cme_month_codes(col))

        ts_df = self._timeseries_builder_helper(start_date=start_date, end_date=end_date, month_codes=list(month_codes), implied_rate=implied_rate)

        cols_to_return = []
        for col in cols:
            try:
                if isinstance(col, tuple):
                    ts_df[col[1]] = ts_df.eval(col[0])
                    cols_to_return.append(col[1])
                else:
                    ts_df[col] = ts_df.eval(col)
                    cols_to_return.append(col)

            except Exception as e:
                self._logger.error(f"'timeseries_builder' eval failed for {col}: {e}")

        return ts_df[cols_to_return].sort_index()

    def continuous_contract_series(self, df: pd.DataFrame, continuous_num: int = 1) -> pd.Series:

        sorted_cols = sorted(df.columns, key=lambda x: self._parse_contract_code(x))

        def get_continuous_value(row: pd.Series):
            valid_values = [row[col] for col in sorted_cols if pd.notna(row[col])]
            if len(valid_values) >= continuous_num:
                return valid_values[continuous_num - 1]
            else:
                return np.nan

        return df.apply(get_continuous_value, axis=1)

    def continuous_timeseries_builder(
        self,
        start_date: datetime,
        end_date: datetime,
        continuous_nums: List[int] = [1],
        implied_rate: Optional[bool] = False,
    ) -> pd.DataFrame:
        cols1 = self._quarterly_contract_codes(start_date.year, num_years=10)
        cols2 = self._quarterly_contract_codes(end_date.year, num_years=10)
        cols = list(set(cols1 + cols2))
        ts_df = self.timeseries_builder(start_date=start_date, end_date=end_date, cols=cols, implied_rate=implied_rate)

        continuous_data = {}
        for num in continuous_nums:
            series = self._continuous_contract_series(ts_df, continuous_num=num)
            continuous_data[f"{num}!"] = series

        continuous_df = pd.DataFrame(continuous_data)
        return continuous_df

    def term_structure_fetcher(self, dates: List[datetime]):
        pass

    def term_structure_plotter(self, dates: List[datetime]):
        pass

    def _extract_cme_month_codes(self, text: str) -> list:
        pattern = r"\b[FGHJKMNQUVXZ]\d{2}\b"
        return re.findall(pattern, text)

    def _quarterly_contract_codes(self, start_year: int, num_years: Optional[int] = 10) -> list:
        quarterly_letters = ["H", "M", "U", "Z"]
        codes = []
        for year in range(start_year, start_year + num_years):
            year_code = str(year)[-2:]
            for letter in quarterly_letters:
                codes.append(f"{letter}{year_code}")
        return codes

    def _parse_contract_code(self, code: str) -> datetime:
        month_code = code[-3:]
        mapping = {"H": 3, "M": 6, "U": 9, "Z": 12}
        letter = month_code[0]
        try:
            year_part = int(month_code[1:])
        except ValueError:
            raise ValueError(f"Invalid year in contract code: {month_code}")
        year = 2000 + year_part
        month = mapping.get(letter)
        if not month:
            raise ValueError(f"Invalid month letter in contract code: {letter}")
        return datetime(year, month, 1)

    def _continuous_contract_series(self, df: pd.DataFrame, continuous_num: int = 1) -> pd.Series:

        sorted_cols = sorted(df.columns, key=lambda x: self._parse_contract_code(x))

        def get_continuous_value(row: pd.Series):
            valid_values = [row[col] for col in sorted_cols if pd.notna(row[col])]
            if len(valid_values) >= continuous_num:
                return valid_values[continuous_num - 1]
            else:
                return np.nan

        return df.apply(get_continuous_value, axis=1)
