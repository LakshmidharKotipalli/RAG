"""Index metadata: what built the index, checked on load to catch mismatches early."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from cfr_rag.config import AppConfig

METADATA_FILE = "metadata.json"


class IndexMismatchError(RuntimeError):
    """The on-disk index was built with a different embedding model or dimension."""


class IndexNotFoundError(RuntimeError):
    """No index exists at the configured path."""


@dataclass
class IndexMetadata:
    embedding_model: str
    embedding_backend: str
    dimension: int
    normalized: bool
    dense_type: str
    lexical_backend: str
    chunker_id: str
    num_chunks: int
    as_of_date: str
    built_at: str

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / METADATA_FILE).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, index_dir: Path) -> IndexMetadata:
        path = Path(index_dir) / METADATA_FILE
        if not path.exists():
            raise IndexNotFoundError(f"No index at {index_dir}. Run `cfr-rag index` first.")
        return cls(**json.loads(path.read_text()))

    def check_compatible(self, cfg: AppConfig, expected_dim: int | None = None) -> None:
        """Raise :class:`IndexMismatchError` if ``cfg`` would embed queries differently."""
        problems = []
        if self.embedding_model != cfg.embedding.model_name:
            problems.append(f"model {self.embedding_model!r} != configured "
                            f"{cfg.embedding.model_name!r}")
        if self.embedding_backend != cfg.embedding.backend:
            problems.append(f"backend {self.embedding_backend!r} != configured "
                            f"{cfg.embedding.backend!r}")
        want = expected_dim if expected_dim is not None else cfg.embedding.dimension
        if want is not None and want != self.dimension:
            problems.append(f"dimension {self.dimension} != configured {want}")
        if problems:
            raise IndexMismatchError(
                "Index/config mismatch: " + "; ".join(problems)
                + ". Rebuild with `cfr-rag index` or point paths.index_dir at a matching index.")
