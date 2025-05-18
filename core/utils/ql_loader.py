import os

# USER CONFIGURABLE: set QL_BACKEND to "quantlib" or "quantlib_risks"

_BACKEND = os.environ.get("QL_BACKEND", "quantlib")

if _BACKEND in ("quantlib_aadc", "aadc"):
    import aadc.quantlib as ql

elif _BACKEND in ("quantlib_risks", "risks"):
    import QuantLib_Risks as ql

elif _BACKEND in ("quantlib",):
    import QuantLib as ql

else:
    print(f"Unknown QuantLib backend '{_BACKEND}'. " "Please set QL_BACKEND to 'quantlib', 'quantlib_risks' or 'quantlib_aadc' ... Using quantlib")
    import QuantLib as ql
