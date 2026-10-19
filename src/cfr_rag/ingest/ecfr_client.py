"""Client for the eCFR versioner API (download full-title XML for a given date)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from cfr_rag.config import ECFRConfig

logger = logging.getLogger(__name__)

TITLE_FILE = "title-{title}.xml"
MANIFEST_FILE = "manifest.json"


class ECFRError(RuntimeError):
    """Raised when the eCFR API cannot be reached or returns an error."""


@dataclass
class SnapshotManifest:
    """Describes one downloaded snapshot of the title."""

    title: int
    as_of_date: str
    mode: str
    source_url: str
    downloaded_at: str
    sha256: str
    bytes: int
    parts: list[str]

    def save(self, directory: Path) -> None:
        (directory / MANIFEST_FILE).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, directory: Path) -> SnapshotManifest:
        return cls(**json.loads((directory / MANIFEST_FILE).read_text()))


class ECFRClient:
    """Thin wrapper over the versioner endpoints with retries and backoff."""

    def __init__(self, cfg: ECFRConfig, session: requests.Session | None = None) -> None:
        self.cfg = cfg
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent, "Accept": "*/*"})

    def _url(self, endpoint: str, **kw: Any) -> str:
        return self.cfg.base_url.rstrip("/") + endpoint.format(title=self.cfg.title, **kw)

    def _get(self, url: str, params: dict[str, Any] | None = None, stream: bool = False) -> requests.Response:
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.cfg.timeout_s, stream=stream)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise ECFRError(f"HTTP {resp.status_code} from {url}")
                resp.raise_for_status()
                return resp
            except (requests.RequestException, ECFRError) as exc:
                last = exc
                if attempt < self.cfg.max_retries:
                    delay = self.cfg.backoff_s * (2**attempt)
                    logger.warning("eCFR request failed (%s); retry %d in %.1fs", exc, attempt + 1, delay)
                    time.sleep(delay)
        raise ECFRError(f"eCFR request failed after {self.cfg.max_retries + 1} attempts: {url}: {last}")

    # ------------------------------------------------------------------ metadata
    def title_info(self) -> dict[str, Any]:
        """Return the titles.json entry for the configured title."""
        data = self._get(self._url(self.cfg.titles_endpoint)).json()
        for entry in data.get("titles", []):
            if int(entry.get("number", -1)) == self.cfg.title:
                return entry
        raise ECFRError(f"Title {self.cfg.title} not found in titles.json")

    def latest_date(self) -> str:
        """The most recent date for which the full title is available."""
        info = self.title_info()
        date = info.get("up_to_date_as_of") or info.get("latest_issue_date")
        if not date:
            raise ECFRError(f"No date available for title {self.cfg.title}: {info}")
        return str(date)

    def structure(self, date: str) -> dict[str, Any]:
        return self._get(self._url(self.cfg.structure_endpoint, date=date)).json()

    def versions_since(self, since: str) -> list[dict[str, Any]]:
        """Section-level change records with issue_date >= ``since`` (YYYY-MM-DD)."""
        data = self._get(self._url(self.cfg.versions_endpoint), params={"issue_date[gte]": since}).json()
        return list(data.get("content_versions", []))

    # ------------------------------------------------------------------ download
    def download(self, raw_dir: Path, date: str | None = None, mode: str | None = None,
                 force: bool = False) -> Path:
        """Download the title for ``date`` (default: latest) into ``raw_dir/<date>/``.

        Returns the snapshot directory. ``mode='full'`` fetches one XML file; if it
        fails (the full title is large and can time out) we fall back to one
        request per part (``mode='by_part'``).
        """
        date = date or self.latest_date()
        mode = mode or self.cfg.download_mode
        out_dir = Path(raw_dir) / date
        if (out_dir / MANIFEST_FILE).exists() and not force:
            logger.info("Snapshot %s already present at %s", date, out_dir)
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if mode == "full":
            try:
                return self._download_full(out_dir, date)
            except ECFRError as exc:
                logger.warning("Full-title download failed (%s); falling back to by_part", exc)
        return self._download_by_part(out_dir, date)

    def _download_full(self, out_dir: Path, date: str) -> Path:
        url = self._url(self.cfg.full_endpoint, date=date)
        logger.info("Downloading %s", url)
        target = out_dir / TITLE_FILE.format(title=self.cfg.title)
        digest, size = _stream_to_file(self._get(url, stream=True), target)
        try:
            (out_dir / "structure.json").write_text(json.dumps(self.structure(date)))
        except ECFRError as exc:
            logger.warning("Could not fetch structure.json (%s); continuing", exc)
        SnapshotManifest(self.cfg.title, date, "full", url, _now(), digest, size, []).save(out_dir)
        logger.info("Saved %s (%.1f MB)", target, size / 1e6)
        return out_dir

    def _download_by_part(self, out_dir: Path, date: str) -> Path:
        from cfr_rag.ingest.parser import parts_from_structure

        structure = self.structure(date)
        (out_dir / "structure.json").write_text(json.dumps(structure))
        parts = parts_from_structure(structure, self.cfg.chapters)
        url = self._url(self.cfg.full_endpoint, date=date)
        hasher, total = hashlib.sha256(), 0
        for i, part in enumerate(parts, 1):
            target = out_dir / f"part-{part}.xml"
            if not target.exists():
                digest, size = _stream_to_file(self._get(url, params={"part": part}, stream=True), target)
            hasher.update(target.read_bytes())
            total += target.stat().st_size
            if i % 50 == 0:
                logger.info("Downloaded %d/%d parts", i, len(parts))
        SnapshotManifest(self.cfg.title, date, "by_part", url, _now(), hasher.hexdigest(), total,
                         parts).save(out_dir)
        return out_dir


def snapshot_source(snapshot_dir: Path, title: int = 21) -> Path:
    """Return the path the parser should read for a snapshot (file or part directory)."""
    full = snapshot_dir / TITLE_FILE.format(title=title)
    return full if full.exists() else snapshot_dir


def latest_snapshot(raw_dir: Path) -> Path | None:
    """Most recent snapshot directory (by as-of date) under ``raw_dir``."""
    dirs = sorted(p for p in Path(raw_dir).glob("*") if (p / MANIFEST_FILE).exists())
    return dirs[-1] if dirs else None


def _stream_to_file(resp: requests.Response, target: Path) -> tuple[str, int]:
    hasher, size = hashlib.sha256(), 0
    tmp = target.with_suffix(target.suffix + ".part")
    with open(tmp, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
    tmp.replace(target)
    return hasher.hexdigest(), size


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
