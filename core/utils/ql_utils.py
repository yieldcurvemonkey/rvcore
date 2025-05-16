import calendar
import functools
from datetime import datetime, timedelta, timezone, time
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from core.utils.ql_loader import ql

DEFAULT_SWAP_TENORS = [
    "1D",
    "1W",
    "2W",
    "3W",
    "1M",
    "2M",
    "3M",
    "4M",
    "5M",
    "6M",
    "9M",
    "12M",
    "18M",
    "2Y",
    "3Y",
    "4Y",
    "5Y",
    "6Y",
    "7Y",
    "8Y",
    "9Y",
    "10Y",
    "12Y",
    "15Y",
    "20Y",
    "25Y",
    "30Y",
    "40Y",
    "50Y",
]


def ql_date_to_datetime(ql_date: ql.Date):
    return datetime(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())


def datetime_to_ql_date(dt: datetime):
    day = dt.day
    month = dt.month
    year = dt.year

    ql_month = {
        1: ql.January,
        2: ql.February,
        3: ql.March,
        4: ql.April,
        5: ql.May,
        6: ql.June,
        7: ql.July,
        8: ql.August,
        9: ql.September,
        10: ql.October,
        11: ql.November,
        12: ql.December,
    }[month]

    return ql.Date(day, ql_month, year)


def tenor_to_ql_period(tenor):
    unit = tenor[-1]
    value = int(tenor[:-1])

    if unit == "D":
        return ql.Period(value, ql.Days)
    elif unit == "W":
        return ql.Period(value, ql.Weeks)
    elif unit == "M":
        return ql.Period(value, ql.Months)
    elif unit == "Y":
        return ql.Period(value, ql.Years)
    else:
        raise ValueError("Invalid tenor unit. Must be one of 'D', 'W', 'M', 'Y'.")


def month_to_label(month: int):
    return f"{month}M" if month <= 12 else f"{int(month / 12)}Y"


def ql_period_to_years(period: ql.Period, day_counter: ql.DayCounter, as_of: datetime):
    end_date = datetime_to_ql_date(as_of) + period
    return day_counter.yearFraction(datetime_to_ql_date(as_of), end_date)


def ql_period_to_months(p: ql.Period) -> int:
    unit = p.units()
    if unit == ql.Years:
        return p.length() * 12
    elif unit == ql.Months:
        return p.length()
    elif unit == ql.Weeks:
        return round(p.length() / 4)
    elif unit == ql.Days:
        return round(p.length() / 30)
    else:
        raise ValueError("Unsupported period unit: {}".format(unit))


def ql_period_to_days(p: ql.Period) -> int:
    unit = p.units()
    if unit == ql.Years:
        return p.length() * 360
    elif unit == ql.Months:
        return p.length() * 30
    elif unit == ql.Weeks:
        return p.length() * 4
    elif unit == ql.Days:
        return p.length()
    else:
        raise ValueError("Unsupported period unit: {}".format(unit))


def parse_frequency(frequency):
    number = int("".join(filter(str.isdigit, frequency)))
    unit = "".join(filter(str.isalpha, frequency))

    if unit == "Y":
        return number, ql.Years
    elif unit == "M":
        return number, ql.Months
    elif unit == "D":
        return number, ql.Days
    elif unit == "W":
        return number, ql.Weeks
    else:
        raise ValueError("Invalid period string format")


def tenor_to_years(tenor):
    num = float(tenor[:-1])
    unit = tenor[-1].upper()
    if unit == "D":
        return num / 360
    if unit == "W":
        return num / 52
    elif unit == "M":
        return num / 12
    elif unit == "Y":
        return num
    else:
        raise ValueError(f"Unknown tenor unit: {tenor}")


def difference_in_ymd(start: ql.Date, end: ql.Date):
    years = end.year() - start.year()
    months = end.month() - start.month()
    days = end.dayOfMonth() - start.dayOfMonth()

    if days < 0:
        # Borrow days from the previous month.
        if end.month() == 1:
            prev_month = 12
            prev_year = end.year() - 1
        else:
            prev_month = end.month() - 1
            prev_year = end.year()
        days_in_prev_month = calendar.monthrange(prev_year, prev_month)[1]
        days += days_in_prev_month
        months -= 1

    if months < 0:
        months += 12
        years -= 1

    return years, months, days


def dates_to_string_tenor(effective_date: datetime, expiration_date: datetime, cal=ql.UnitedStates(ql.UnitedStates.GovernmentBond)):
    """
    Returns a tenor string (e.g. "6Y" or "5Y3M10D") between effective_date and expiration_date,
    after adjusting each date to a business day using the given QuantLib calendar.

    Rounding rules:
      - If days > 25, treat that as an additional month (i.e. add 1 to months, set days to 0).
      - If the (possibly updated) months count is >= 11, treat that as an additional year.

    If no nonzero component exists, returns "Spot".
    """
    try:
        eff = ql.Date(effective_date.day, effective_date.month, effective_date.year)
        exp = ql.Date(expiration_date.day, expiration_date.month, expiration_date.year)

        if not cal.isBusinessDay(eff):
            eff = cal.adjust(eff, ql.Following)
        if not cal.isBusinessDay(exp):
            exp = cal.adjust(exp, ql.Following)

        years, months, days = difference_in_ymd(eff, exp)

        if days > 15:
            months += 1
            days = 0

        if months >= 11:
            years += 1
            months = 0

        tenor = ""
        if years:
            tenor += f"{years}Y"
        if months:
            tenor += f"{months}M"
        if days > 15:
            tenor += f"{days}D"

        if tenor:
            return tenor
        return "0D"

    except Exception as e:
        # print(e, effective_date, expiration_date)
        return "NaN"


def get_bdates_between(start_date: datetime, end_date: datetime, calendar: ql.Calendar) -> List[datetime]:
    bdates = calendar.businessDayList(datetime_to_ql_date(start_date), datetime_to_ql_date(end_date))
    return sorted([ql_date_to_datetime(bd) for bd in bdates])


def datetime_today_utc():
    return datetime(
        year=datetime.now(timezone.utc).year,
        month=datetime.now(timezone.utc).month,
        day=datetime.now(timezone.utc).day,
        hour=0,
        minute=0,
    )


def period_to_string(period: ql.Period):
    if period.units() == ql.Months:
        return f"{int(period.length())}M"
    elif period.units() == ql.Years:
        return f"{int(period.length())}Y"
    else:
        raise ValueError("Unsupported period unit.")


def period_to_months(period: ql.Period):
    if period.units() == ql.Months:
        return period.length()
    elif period.units() == ql.Years:
        return period.length() * 12
    else:
        raise ValueError("Unsupported period unit.")


def parse_tenor_string(tenor_str):
    if tenor_str.endswith("M"):
        return ql.Period(int(tenor_str[:-1]), ql.Months)
    elif tenor_str.endswith("Y"):
        return ql.Period(int(tenor_str[:-1]), ql.Years)
    else:
        raise ValueError("Unsupported tenor format: " + tenor_str)


def most_recent_business_day_ql(ql_calendar: ql.Calendar, tz: Optional[str] = "UTC", to_pydate: Optional[bool] = False):
    current_ts = pd.Timestamp.now(ZoneInfo(tz)).normalize()
    current_pydate = current_ts.to_pydatetime().date()
    current_ql = ql.Date(current_pydate.day, current_pydate.month, current_pydate.year)

    while not ql_calendar.isBusinessDay(current_ql):
        current_ql = current_ql - 1

    if to_pydate:
        return datetime(current_ql.year(), current_ql.month(), current_ql.dayOfMonth())
    else:
        return current_ql


def most_recent_business_day_from_date(input_date: datetime, ql_calendar: ql.Calendar, tz: str = "UTC", to_pydate: bool = True):
    ts = pd.Timestamp(input_date, tz=tz).normalize()
    pydate = ts.to_pydatetime().date()
    current_ql = ql.Date(pydate.day, pydate.month, pydate.year)

    while not ql_calendar.isBusinessDay(current_ql):
        current_ql = current_ql - 1  # subtract one day

    if to_pydate:
        return datetime(current_ql.year(), current_ql.month(), current_ql.dayOfMonth())
    else:
        return current_ql


@functools.lru_cache(maxsize=None)
def _period_from_str(period_str: str) -> ql.Period:
    return ql.Period(period_str)
