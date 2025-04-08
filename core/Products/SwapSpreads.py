import re
import warnings
from datetime import datetime
from typing import List, Optional

import pandas as pd
import QuantLib as ql

from core.Products.BaseProductPlotter import BaseProductPlotter
from core.Products.Cash import Cash
from core.Products.Swaps import Swaps
from core.utils.ql_utils import ql_period_to_months


class SwapSpreads(BaseProductPlotter):
    _swap_product: Swaps = None
    _cash_product: Cash = None
    _default_tenors: List[str] = None
    _DEFAULT_BENCHMARK_SPREAD_PREFIX = "SS"
    _DEFAULT_MATCHED_MATURITY_SWAP_SPREAD_PREFIX = "MMSS"
    _default_tz: str = "US/Eastern"
    _MAX_NJOBS = None

    def __init__(self, swap_product: Swaps, cash_product: Cash, default_swap_tenors: Optional[List[str]] = None, max_njobs: Optional[int] = 1):
        self._swap_product = swap_product
        self._cash_product = cash_product
        if default_swap_tenors:
            self._default_tenors = default_swap_tenors
        else:
            self._default_tenors = [
                f"{self._swap_product._DEFAULT_SWAP_PREFIX}_0Dx1Y",
                f"{self._swap_product._DEFAULT_SWAP_PREFIX}_0Dx2Y",
                f"{self._swap_product._DEFAULT_SWAP_PREFIX}_0Dx3Y",
                f"{self._swap_product._DEFAULT_SWAP_PREFIX}_0Dx5Y",
                f"{self._swap_product._DEFAULT_SWAP_PREFIX}_0Dx7Y",
                f"{self._swap_product._DEFAULT_SWAP_PREFIX}_0Dx10Y",
                f"{self._swap_product._DEFAULT_SWAP_PREFIX}_0Dx20Y",
                f"{self._swap_product._DEFAULT_SWAP_PREFIX}_0Dx30Y",
            ]

        self._MAX_NJOBS = max_njobs

    def benchmark_spreads_timeseries_builder(self, start_date: datetime, end_date: datetime, cols: Optional[List[str]] = None):
        swap_ts_df = self._swap_product.swaps_timeseries_builder(start_date=start_date, end_date=end_date, cols=self._default_tenors, n_jobs=self._MAX_NJOBS)
        cash_ts_df = self._cash_product.otr_timeseries_builder(
            start_date=start_date,
            end_date=end_date,
            cols=self._cash_product._benchmark_tenors,
        )

        cash_ts_df["cash_timestamp"] = cash_ts_df.index
        swap_ts_df["date"] = swap_ts_df.index.normalize()
        cash_ts_df["date"] = cash_ts_df.index.normalize()
        benchmark_spreads_cols = zip(self._default_tenors, self._cash_product._benchmark_tenors)
        benchmark_spreads_df = pd.merge(swap_ts_df, cash_ts_df, on="date", how="inner", suffixes=("_usd", "_otr"))
        benchmark_spreads_df.set_index("cash_timestamp", inplace=True)

        for spread in benchmark_spreads_cols:
            first_col, second_col = spread
            benchmark_spreads_df[f"{self._DEFAULT_BENCHMARK_SPREAD_PREFIX}_{first_col.split("x")[1]}"] = benchmark_spreads_df.eval(
                f"(`{first_col}` - `{second_col}`) * 100"
            )
            benchmark_spreads_df = benchmark_spreads_df.drop(columns=[first_col, second_col])

        benchmark_spreads_df.drop(columns=["date"], inplace=True)
        benchmark_spreads_df.index.name = self._cash_product._timestamp_col

        if cols:
            cols_to_return = []
            for col in cols:
                try:
                    if isinstance(col, tuple):
                        benchmark_spreads_df[col[1]] = benchmark_spreads_df.eval(col[0])
                        cols_to_return.append(col[1])
                    else:
                        benchmark_spreads_df[col] = benchmark_spreads_df.eval(col)
                        cols_to_return.append(col)
                except Exception as e:
                    self._logger.error(f"'timeseries_builder' eval failed for {col}: {e}")

            benchmark_spreads_df = benchmark_spreads_df[cols_to_return].sort_index()

        return benchmark_spreads_df

    def _extract_cusips(self, text: str, prefix: str) -> List[str]:
        escaped_prefix = re.escape(prefix)
        pattern = rf"(?i)(?<={escaped_prefix}_)[0-9A-Z]{{9}}\b"
        cusips = re.findall(pattern, text)
        return cusips

    # TOOO make data fetching data `cusip_timeseries_builder` is the bottleneck
    def maturity_matched_spreads_timeseries_builder(
        self, start_date: datetime, end_date: datetime, cols: List[str], ignore_eval: Optional[bool] = False, ytm_type: Optional[str] = "eod_ytm"
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            cusips = []
            for col in cols:
                if isinstance(col, tuple):
                    col = col[0]
                cusips += self._extract_cusips(col, self._DEFAULT_MATCHED_MATURITY_SWAP_SPREAD_PREFIX)

            if not cusips:
                raise ValueError("CUSIP Extraction Failed")

            cusip_set_df = self._cash_product.cusip_curve_set_builder(start_date=end_date, end_date=end_date)[end_date]
            cusip_set_df = cusip_set_df[cusip_set_df["cusip"].isin(cusips)]

            cusip_mats: List[pd.Timestamp] = zip(cusip_set_df["cusip"].to_list(), cusip_set_df["maturity_date"].to_list())
            mms_tenors = [
                (f"{self._swap_product._DEFAULT_MATURITY_MATCHED_PREFIX}_{mat.strftime("%b_%d_%Y")}", f"{self._DEFAULT_MATCHED_MATURITY_SWAP_SPREAD_PREFIX}_{cusip}")
                for cusip, mat in cusip_mats
            ]

            mms_ts_df = self._swap_product.swaps_timeseries_builder(start_date=start_date, end_date=end_date, cols=mms_tenors, n_jobs=self._MAX_NJOBS)
            cusip_ts_df = self._cash_product.cusip_timeseries_builder(start_date=start_date, end_date=end_date, cusips=cusips, cols_to_return=[ytm_type])

            def _mms_subtract_dfs_helper(swaps_df: pd.DataFrame, cash_df: pd.DataFrame) -> pd.DataFrame:
                swaps = swaps_df.copy()
                cash = cash_df.copy()
                cash_full_ts = cash.index.copy()
                swaps_full_ts = swaps.index.copy()

                swaps.index = pd.to_datetime(swaps.index).date
                cash.index = pd.to_datetime(cash.index).date

                result = cash.join(swaps, how="inner")

                swaps_prefix = {col.split("_")[1]: col for col in swaps.columns}
                cash_prefix = {col.split("_")[0]: col for col in cash.columns}

                to_return_result_cols = []
                for prefix, swap_col in swaps_prefix.items():
                    if prefix in cash_prefix:
                        cash_col = cash_prefix[prefix]
                        result[swap_col] = swaps[swap_col] - cash[cash_col]
                        to_return_result_cols.append(swap_col)
                    else:
                        print(f"Warning: No matching cash column for cusip prefix {prefix}.")

                try:
                    result.index = cash_full_ts
                    result.index = result.index.tz_convert(self._default_tz)
                except Exception as e:
                    self._logger.error(f"'maturity_matched_spreads_timeseries_builder' failed to set index: {e}")

                    result.index = swaps_full_ts
                    result.index = result.index.tz_convert(self._default_tz)

                return result[to_return_result_cols]

            ts_df = _mms_subtract_dfs_helper(mms_ts_df, cusip_ts_df)

            if ignore_eval:
                return ts_df.sort_index()

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

    def spreads_term_structure_plotter(self, dates: List[datetime], benchmark_spreads_df: Optional[pd.DataFrame] = None):
        # TODO we are fetching way too much data
        benchmark_spreads_df = benchmark_spreads_df or self.benchmark_spreads_timeseries_builder(start_date=min(dates), end_date=max(dates))
        dates = [dt.date() for dt in dates]
        benchmark_spreads_dict_df = {dt: benchmark_spreads_df.loc[[dt]] for dt in benchmark_spreads_df.index if dt.date() in dates}
        benchmark_spreads_dict_df
        self._swap_product._term_structure_plotter(
            term_structure_dict_df=benchmark_spreads_dict_df,
            plot_title="Benchmark Swap Spreads",
            x_axis_col_sorter_func=lambda x: ql_period_to_months(ql.Period(x)),
            use_plotly=True,
        )
