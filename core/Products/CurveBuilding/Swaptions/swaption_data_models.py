import warnings
from typing import ClassVar

from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SAWarning
from sqlalchemy.ext.declarative import declarative_base, declared_attr

Base = declarative_base()


class SCubeCacheBase(Base):
    __abstract__ = True
    CURVE: ClassVar[str] = None

    @declared_attr
    def __tablename__(cls):
        if not cls.CURVE:
            raise ValueError("CURVE must be set for SCubeCacheBase subclasses")
        return f"{cls.CURVE}_SCube"

    timestamp = Column(String, primary_key=True)
    scube = Column(JSONB)


class SABRParamsCacheBase(Base):
    __abstract__ = True
    CURVE: ClassVar[str] = None

    @declared_attr
    def __tablename__(cls):
        if not cls.CURVE:
            raise ValueError("CURVE must be set for SABRParamsCacheBase subclasses")
        return f"{cls.CURVE}_SABRParams"

    timestamp = Column(String, primary_key=True)
    sabr_params = Column(JSONB)


def get_scube_cache_model(curve: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SAWarning)

        class SCubeCache(SCubeCacheBase):
            __table_args__ = {"extend_existing": True}
            CURVE = curve

        return SCubeCache


def get_sabr_params_cache_model(curve: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SAWarning)

        class SABRParamsCache(SABRParamsCacheBase):
            __table_args__ = {"extend_existing": True}
            CURVE = curve

        return SABRParamsCache
