"""High-level ingestion operations used by the CLI and scripts."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

from cfr_rag.config import AppConfig
from cfr_rag.ingest.diff import SnapshotDiff, diff_snapshots
from cfr_rag.ingest.ecfr_client import (
    ECFRClient,
    SnapshotManifest,
    latest_snapshot,
    snapshot_source,
)
from cfr_rag.ingest.models import Section
from cfr_rag.ingest.parser import ECFRParser
from cfr_rag.store.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


def mark_definitions(sections: Iterable[Section], heading_pattern: str) -> list[Section]:
    """Flag sections whose heading looks like a definitions section."""
    pat = re.compile(heading_pattern)
    out = []
    for s in sections:
        s.is_definitions = bool(pat.search(s.heading)) and not s.reserved
        out.append(s)
    return out


def parse_snapshot(cfg: AppConfig, snapshot_dir: Path) -> list[Section]:
    """Parse a downloaded snapshot directory into sections."""
    manifest = SnapshotManifest.load(snapshot_dir)
    parser = ECFRParser(chapters=cfg.ecfr.chapters, as_of_date=manifest.as_of_date,
                        title=cfg.ecfr.title)
    sections = list(parser.parse_path(snapshot_source(snapshot_dir, cfg.ecfr.title)))
    sections = mark_definitions(sections, cfg.chunking.definitions.heading_pattern)
    logger.info("Parsed %d sections (%d reserved, %d definitions) from %s", len(sections),
                sum(s.reserved for s in sections), sum(s.is_definitions for s in sections),
                snapshot_dir)
    return sections


def ingest(cfg: AppConfig, date: str | None = None, xml_path: Path | None = None,
           force: bool = False) -> list[Section]:
    """Download (unless ``xml_path`` is given), parse and store the title.

    Args:
        date: eCFR as-of date (YYYY-MM-DD); default is the latest available.
        xml_path: Parse a local XML file instead of downloading (offline use).
        force: Re-download even if the snapshot already exists.
    """
    if xml_path:
        as_of = date or "local"
        parser = ECFRParser(chapters=cfg.ecfr.chapters, as_of_date=as_of, title=cfg.ecfr.title)
        sections = mark_definitions(parser.parse_path(xml_path),
                                    cfg.chunking.definitions.heading_pattern)
        source = str(xml_path)
    else:
        snap = ECFRClient(cfg.ecfr).download(cfg.paths.raw_dir, date=date, force=force)
        sections = parse_snapshot(cfg, snap)
        as_of = SnapshotManifest.load(snap).as_of_date
        source = str(snap)
    store = SQLiteStore(cfg.paths.db_path)
    try:
        n = store.replace_sections(sections)
        store.set_meta("as_of_date", as_of)
        store.set_meta("source", source)
        store.set_meta("section_hashes_at_ingest", store.section_hashes())
    finally:
        store.close()
    logger.info("Stored %d sections as of %s in %s", n, as_of, cfg.paths.db_path)
    return sections


def check_updates(cfg: AppConfig, new_date: str | None = None,
                  old_snapshot: Path | None = None) -> tuple[SnapshotDiff, list[dict]]:
    """Download a newer snapshot and diff it against the stored (or given) one.

    Does not modify the database. Returns the diff and the versioner API's
    change records since the old date (empty if that call fails).
    """
    client = ECFRClient(cfg.ecfr)
    store = SQLiteStore(cfg.paths.db_path)
    try:
        if old_snapshot:
            old_sections = parse_snapshot(cfg, old_snapshot)
            old_date = SnapshotManifest.load(old_snapshot).as_of_date
        else:
            old_sections = list(store.iter_sections())
            old_date = store.get_meta("as_of_date", "")
    finally:
        store.close()
    if not old_sections:
        snap = latest_snapshot(cfg.paths.raw_dir)
        if snap is None:
            raise RuntimeError("Nothing to compare against: run `cfr-rag ingest` first.")
        old_sections, old_date = parse_snapshot(cfg, snap), SnapshotManifest.load(snap).as_of_date
    new_snap = client.download(cfg.paths.raw_dir, date=new_date)
    new_sections = parse_snapshot(cfg, new_snap)
    new_as_of = SnapshotManifest.load(new_snap).as_of_date
    diff = diff_snapshots(old_sections, new_sections, old_date, new_as_of)
    versions: list[dict] = []
    if old_date and old_date != "local":
        try:
            versions = client.versions_since(old_date)
        except Exception as exc:  # the report is still useful without it
            logger.warning("Could not fetch versioner change records: %s", exc)
    return diff, versions
