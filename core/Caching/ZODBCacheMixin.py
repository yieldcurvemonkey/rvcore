import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple

import transaction
from BTrees.OOBTree import OOBTree
from persistent.mapping import PersistentMapping
from zc.lockfile import LockError
from ZODB import DB, Connection, DemoStorage, FileStorage

from core.Caching.CodecMapping import CodecMapping

EncodeFn = Callable[[Any], Any]
DecodeFn = Callable[[Any], Any]


class ZODBCacheMixin:
    def __init__(self, use_btree: Optional[bool] = True, *args, **kwargs):
        self._use_btree = use_btree
        super().__init__(*args, **kwargs)
        self._zconn_map: Dict[str, Tuple[DB, FileStorage.FileStorage]] = {}
        self._codec_wrappers: Dict[str, CodecMapping] = {}

    @staticmethod
    def _slug(text: str, repl: str = "_") -> str:
        return re.sub(r"[^\w.\-]", repl, text)

    @staticmethod
    def default_cache_path(stem: str, ext: str = ".fs") -> str:
        safe = ZODBCacheMixin._slug(stem)
        root = Path(os.getenv("ZODB_CACHE_DIR", Path(__file__).resolve().parent))
        root.mkdir(parents=True, exist_ok=True)
        return str(root / "dump" / f"{safe}{ext}")

    def _safe_storage(self, path: str) -> FileStorage.FileStorage:
        try:
            return FileStorage.FileStorage(path)
        except LockError:
            ro = FileStorage.FileStorage(path, read_only=True)
            return DemoStorage(base=ro)

    def zodb_open_cache(
        self,
        *,
        cache_attr: str,
        path: str,
        encode: Optional[EncodeFn] = None,
        decode: Optional[DecodeFn] = None,
    ) -> None:
        if cache_attr in self._zconn_map:
            return

        try:
            storage = FileStorage.FileStorage(path)
        except LockError:
            for attr, (conn, stor) in list(self._zconn_map.items()):
                if getattr(stor, "_file_name", None) == path:
                    conn.close()
                    stor.close()
                    del self._zconn_map[attr]

            try:
                storage = FileStorage.FileStorage(path)
            except LockError:
                storage = self._safe_storage(path)

        db = DB(storage)
        try:
            conn: Connection = db.open(transaction_manager=transaction.manager)
        except TypeError:
            conn: Connection = db.open(txn_manager=transaction.manager)
        # ensure a persistent transaction manager is attached
        conn.transaction_manager = transaction.manager
        root = conn.root()

        if cache_attr not in root:
            container = OOBTree() if getattr(self, "_use_btree", True) else PersistentMapping()
            root[cache_attr] = container
            transaction.commit()

        mapping: MutableMapping = root[cache_attr]
        if encode or decode:
            mapping = CodecMapping(mapping, encode, decode)

        setattr(self, cache_attr, mapping)
        self._zconn_map[cache_attr] = (conn, storage)

    def zodb_commit(self) -> None:
        transaction.commit()

    def close_zodb(self) -> None:
        for conn, storage in self._zconn_map.values():
            conn.close()
            storage.close()
        self._zconn_map.clear()
        self._codec_wrappers.clear()
