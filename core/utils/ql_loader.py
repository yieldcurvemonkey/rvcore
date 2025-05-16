import os

# USER CONFIGURABLE: set QL_BACKEND to "quantlib" or "quantlib_risks"
_BACKEND = os.environ.get("QL_BACKEND", "quantlib")
# _BACKEND = os.environ.get("QL_BACKEND", "quantlib_risks")

print("USING BACKEND: ", _BACKEND)

if _BACKEND in ("quantlib_risks", "risks"):
    import QuantLib_Risks as ql

elif _BACKEND in ("quantlib",):
    import QuantLib as ql

else:
    raise ImportError(f"Unknown QuantLib backend '{_BACKEND}'. " "Please set QL_BACKEND to 'quantlib' or 'quantlib_risks'.")
