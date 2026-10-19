import json
from pathlib import Path

from conftest import FIXTURE_XML

from cfr_rag.config import load_config
from cfr_rag.ingest.diff import diff_snapshots
from cfr_rag.ingest.ecfr_client import ECFRClient
from cfr_rag.ingest.parser import ECFRParser
from cfr_rag.ingest.service import ingest
from cfr_rag.store.sqlite_store import SQLiteStore


def cfg_for(tmp_path: Path):
    return load_config("config/default.yaml", overrides={"paths": {
        "data_dir": str(tmp_path), "raw_dir": str(tmp_path / "raw"),
        "db_path": str(tmp_path / "db.sqlite"), "index_dir": str(tmp_path / "index"),
        "cache_dir": str(tmp_path / "cache")}})


def test_ingest_local_xml_roundtrip(tmp_path: Path):
    cfg = cfg_for(tmp_path)
    secs = ingest(cfg, date="2026-10-01", xml_path=FIXTURE_XML)
    store = SQLiteStore(cfg.paths.db_path)
    assert store.count_sections() == len(secs)
    s = store.get_section("211.68")
    assert s is not None and s.paragraphs[1].path == "(b)"
    assert store.get_meta("as_of_date") == "2026-10-01"


def test_diff_detects_changes():
    old = list(ECFRParser().parse_path(FIXTURE_XML))
    new = list(ECFRParser().parse_path(FIXTURE_XML))
    new[0].paragraphs[0].text += " Amended."
    removed = new.pop()
    diff = diff_snapshots(old, new, "a", "b")
    assert diff.modified == [old[0].section_id]
    assert diff.removed == [removed.section_id]
    assert "Amended." in diff.details[old[0].section_id]
    assert "Modified: **1**" in diff.to_markdown()


class _Resp:
    def __init__(self, payload=None, content=b"", status=200):
        self.payload, self.content, self.status_code = payload, content, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=0):
        yield self.content


class _Session:
    def __init__(self):
        self.headers, self.calls = {}, []

    def get(self, url, params=None, timeout=None, stream=False):
        self.calls.append((url, params))
        if url.endswith("titles.json"):
            return _Resp({"titles": [{"number": 21, "up_to_date_as_of": "2026-10-15"}]})
        if "structure" in url:
            return _Resp({"type": "title", "children": []})
        return _Resp(content=FIXTURE_XML.read_bytes())


def test_client_downloads_latest_with_manifest(tmp_path: Path):
    cfg = cfg_for(tmp_path)
    client = ECFRClient(cfg.ecfr, session=_Session())
    snap = client.download(cfg.paths.raw_dir)
    manifest = json.loads((snap / "manifest.json").read_text())
    assert manifest["as_of_date"] == "2026-10-15" and manifest["mode"] == "full"
    assert (snap / "title-21.xml").stat().st_size > 1000
    assert client.session.calls[1][0].endswith("/full/2026-10-15/title-21.xml")
