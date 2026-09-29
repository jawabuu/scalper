"""
The version string is the ONLY thing that tells an operator which code is
running. A stale one is worse than none: it is affirmatively misleading.

Versions 3.95.0, 3.95.1 and 3.96.0 were all shipped declaring "3.94.2"
because a sed pattern stopped matching and failed SILENTLY. The banner said
3.94.2 while three fixes were live, and a conclusion was drawn from that
banner ("the arm gate must already be in your build") that was wrong.
"""

import re
from pathlib import Path

import bot


ROOT = Path(__file__).resolve().parents[1]


def test_the_version_is_a_plain_three_part_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", bot.__version__), bot.__version__


def test_the_version_appears_exactly_once_in_the_package():
    """
    One definition, so a bump cannot half-apply. If this ever fails, some
    other file has started carrying its own copy and the two can disagree.
    """
    hits = []
    for p in (ROOT / "bot").rglob("*.py"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("__version__"):
                hits.append(f"{p.name}: {line}")
    assert len(hits) == 1, hits


def test_the_handover_header_matches_the_code():
    """
    HANDOVER.md leads with the version it documents. When the two disagree,
    one of them is lying to whoever reads it next — and the handover is what
    a future session trusts.
    """
    head = (ROOT / "HANDOVER.md").read_text(encoding="utf-8")[:4000]
    m = re.search(r"\*\*v(\d+\.\d+\.\d+)\*\*", head)
    assert m, "no **vX.Y.Z** in the handover header"
    assert m.group(1) == bot.__version__, (
        f"handover says {m.group(1)}, code says {bot.__version__}")
