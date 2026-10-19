"""Inspect a downloaded eCFR XML file before trusting the parser.

Prints DIV TYPE counts, the tags that occur inside sections, tags the parser
does not explicitly handle, and one sample parsed section.

Usage:
    python scripts/inspect_xml.py data/raw/<date>/title-21.xml [--section 211.68]
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from lxml import etree

from cfr_rag.ingest.parser import (
    BLOCK_TAGS,
    HIERARCHY_TYPES,
    ITALIC_TAGS,
    LEAF_TYPES,
    NOTE_TAGS,
    SKIP_TAGS,
    ECFRParser,
)

KNOWN = (BLOCK_TAGS | ITALIC_TAGS | NOTE_TAGS | SKIP_TAGS
         | {"GPOTABLE", "EXTRACT", "SU", "FTNT", "HED", "BOXHD", "CHED", "ROW", "ENT", "TTITLE",
            "APPRO", "DIV"})

logger = logging.getLogger("inspect_xml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--section", default="211.68")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tree = etree.parse(args.xml, etree.XMLParser(huge_tree=True, recover=True))
    root = tree.getroot()
    div_types: Counter[str] = Counter()
    in_section: Counter[str] = Counter()
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if el.tag.startswith("DIV"):
            div_types[f"{el.tag} TYPE={el.get('TYPE')}"] += 1
        if el.tag.startswith("DIV") and el.get("TYPE") in LEAF_TYPES:
            for child in el.iter():
                if isinstance(child.tag, str) and child is not el:
                    in_section[child.tag] += 1
    logger.info("Root: <%s>", root.tag)
    logger.info("\nDIV types:")
    for k, v in sorted(div_types.items()):
        flag = "" if (k.split("TYPE=")[1] in HIERARCHY_TYPES | LEAF_TYPES) else "   <-- not handled"
        logger.info("  %-28s %7d%s", k, v, flag)
    logger.info("\nTags inside sections/appendices:")
    for tag, n in in_section.most_common():
        flag = "" if tag in KNOWN else "   <-- not explicitly handled (text still flattened)"
        logger.info("  %-12s %8d%s", tag, n, flag)
    secs = {s.section_id: s for s in ECFRParser().parse_path(args.xml)}
    logger.info("\nParsed %d sections; %d with cross-refs; %d with defined terms", len(secs),
                sum(bool(s.cross_refs) for s in secs.values()),
                sum(any(p.term for p in s.paragraphs) for s in secs.values()))
    s = secs.get(args.section)
    if s:
        logger.info("\nSample § %s %s (chapter %s, part %s, subpart %s)", s.section_id, s.heading,
                    s.chapter, s.part, s.subpart)
        for p in s.paragraphs[:12]:
            logger.info("  %-14s %s", p.path or "-", p.text[:90])
        logger.info("  cross_refs: %s", s.cross_refs)
        logger.info("  amendments: %s", s.source_citation[:200])


if __name__ == "__main__":
    main()
