"""Unit tests for data/corpus.py + data/accessibility.py against the mini fixtures.

The fixtures and their hand-derived expected values are in tests/fixtures/EXPECTED.md. These tests
assert the PRODUCTION loaders reproduce that contract (the same contract the Phase 3 smoke oracle
proved), covering: locate_premise position-containment, transitive imports, same-file position
accessibility, and import isolation.

Premise UID = "{path}::{full_name}@{start_line},{start_col}" (collision-free; see corpus.py).
"""

from __future__ import annotations

from pathlib import Path

from prooflens.data.accessibility import accessible_premises
from prooflens.data.corpus import Premise, load_corpus

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"

A = "Mathlib/Algebra/Basic.lean"
B = "Mathlib/Order/Basic.lean"
C = "Mathlib/Topology/Basic.lean"

# expected UIDs (all fixture premises start at column 1)
ADD = f"{A}::add_comm@10,1"
MUL = f"{A}::mul_comm@20,1"
LEREFL = f"{B}::le_refl@5,1"
LETRANS = f"{B}::le_trans@15,1"
ISOPEN = f"{C}::isOpen_univ@8,1"
CONT = f"{C}::continuous_id@25,1"


def _corpus():
    return load_corpus(str(FIX / "corpus.jsonl"))


# -- corpus loading ---------------------------------------------------------------------------

def test_corpus_loads_all_premises():
    c = _corpus()
    assert len(c) == 6
    assert set(c.paths) == {A, B, C}
    assert {p.uid for p in c.get_premises(A)} == {ADD, MUL}
    # fields round-tripped
    add_comm = next(p for p in c.get_premises(A) if p.full_name == "add_comm")
    assert add_comm.start == (10, 1) and add_comm.end == (12, 40)
    assert add_comm.kind == "lemma"
    assert add_comm.uid == ADD


def test_premise_identity_is_path_name_start():
    p1 = Premise(A, "add_comm", (10, 1), (12, 40), "code1", "lemma")
    p2 = Premise(A, "add_comm", (10, 1), (99, 99), "code2", "def")   # differ only in end/code/kind
    p3 = Premise(A, "add_comm", (11, 1), (12, 40), "code1", "lemma")  # different start
    assert p1 == p2 and hash(p1) == hash(p2)      # end/code/kind excluded from identity
    assert p1 != p3                                # start is part of identity


# -- locate_premise (position containment) ----------------------------------------------------

def test_locate_premise_containment():
    c = _corpus()
    # continuous_id spans [25,1]-[27,45]; def_pos [25,9] is inside
    p = c.locate_premise(C, (25, 9))
    assert p is not None and p.full_name == "continuous_id"
    # exact start and exact end are inclusive
    assert c.locate_premise(C, (25, 1)).full_name == "continuous_id"
    assert c.locate_premise(C, (27, 45)).full_name == "continuous_id"


def test_locate_premise_miss_returns_none():
    c = _corpus()
    # line 99 in Algebra is not spanned by any premise -> None (the dropped `foo_missing` case)
    assert c.locate_premise(A, (99, 3)) is None
    # a position in a gap between premises
    assert c.locate_premise(A, (15, 1)) is None
    # unknown file
    assert c.locate_premise("Does/Not/Exist.lean", (1, 1)) is None


# -- transitive imports -----------------------------------------------------------------------

def test_transitive_imports():
    c = _corpus()
    assert c.transitive_imports(C) == {B, A}     # Topology -> Order -> Algebra
    assert c.transitive_imports(B) == {A}
    assert c.transitive_imports(A) == set()


# -- accessibility ----------------------------------------------------------------------------

def test_accessibility_topology_theorem_sees_all():
    c = _corpus()
    # theorem in Topology at start [40,1]: all 6 premises accessible
    acc = accessible_premises(c, C, (40, 1))
    assert acc == {ADD, MUL, LEREFL, LETRANS, ISOPEN, CONT}


def test_accessibility_order_theorem_excludes_topology():
    c = _corpus()
    # theorem in Order at start [30,1]: A (imported) + B same-file earlier; NOT Topology
    acc = accessible_premises(c, B, (30, 1))
    assert acc == {ADD, MUL, LEREFL, LETRANS}
    assert CONT not in acc and ISOPEN not in acc


def test_accessibility_same_file_position_boundary():
    c = _corpus()
    # In Order at a position BEFORE le_trans ends: le_trans (end [17,50]) not yet accessible,
    # le_refl (end [6,30]) is. Imported Algebra premises are always accessible.
    acc = accessible_premises(c, B, (7, 0))
    assert LEREFL in acc          # end [6,30] <= [7,0]
    assert LETRANS not in acc     # end [17,50] > [7,0]
    assert ADD in acc             # imported -> always accessible
