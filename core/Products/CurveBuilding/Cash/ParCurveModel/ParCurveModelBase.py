import abc
import logging
from typing import Callable, Any, Dict

import QuantLib as ql
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class ParCurveModelBase(metaclass=abc.ABCMeta):
    _TO_RETURN_TTM_COL = "time_to_maturity"
    _TO_RETURN_DATE_COL = "date"
    _TO_RETURN_DF_COL = "discount_factor"
    _TO_RETURN_ZERO_COL = "zero_rate"
    _TO_RETURN_PAR_COL = "par_rate"

    _FILTER_TOO_STRONG = "no bonds left after filtering!"

    _cusip_set_df: pd.DataFrame = None
    _filtered_cusip_set_df: pd.DataFrame = None
    _fitted_curve_df: pd.DataFrame = None
    _ql_discount_curve: ql.DiscountCurve 
    _scipy_par_spline_func: Callable[[np.number], np.number] | interp1d = None
    _optimization_results: Any = None
    _arb_free_results: Dict

    def __init__(
        self,
        debug_verbose: bool = False,
        info_verbose: bool = False,
        warning_verbose: bool = False,
        error_verbose: bool = False,
    ):
        self._debug_verbose = debug_verbose
        self._info_verbose = info_verbose
        self._error_verbose = error_verbose
        self._warning_verbose = warning_verbose
        self._setup_logger()

    def _setup_logger(self):
        self._logger = logging.getLogger(self.__class__.__name__)

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(handler)

        if self._debug_verbose:
            self._logger.setLevel(logging.DEBUG)
        elif self._info_verbose:
            self._logger.setLevel(logging.INFO)
        elif self._error_verbose:
            self._logger.setLevel(logging.ERROR)
        elif self._warning_verbose:
            self._logger.setLevel(logging.WARNING)
        else:
            self._logger.disabled = True

    @abc.abstractmethod
    def _filter(cusip_set_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def fit(cusip_set_df: pd.DataFrame, kwags):
        pass
