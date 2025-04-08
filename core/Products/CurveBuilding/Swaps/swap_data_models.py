import warnings
from typing import ClassVar

from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import HSTORE
from sqlalchemy.exc import SAWarning
from sqlalchemy.ext.declarative import declarative_base, declared_attr

Base = declarative_base()


class QLCurveCacheBase(Base):
    __abstract__ = True
    CURVE: ClassVar[str] = None
    INTERPOLATION: ClassVar[str] = None

    @declared_attr
    def __tablename__(cls):
        if not cls.CURVE or not cls.INTERPOLATION:
            raise ValueError("CURVE and INTERPOLATION must be set on concrete QLCurveCache classes")
        return f"{cls.CURVE}_{cls.INTERPOLATION}_ql_cache_nodes"

    timestamp = Column(String, primary_key=True)
    nodes = Column(HSTORE)


def get_ql_curve_cache_model(curve: str, interpolation: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SAWarning)

        class QLCurveCache(QLCurveCacheBase):
            __table_args__ = {"extend_existing": True}
            CURVE = curve
            INTERPOLATION = interpolation

        return QLCurveCache
