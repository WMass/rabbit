"""The saturated fit must not snapshot over the main fit's snapshot.

``save_hists`` builds the saturated fitter with ``copy.deepcopy(fitter)``, so it
inherits ``--snapshotFile``. Its own ``minimize()`` then writes periodic
snapshots, and without a separate path those land on the main fit's file.

That is not a cosmetic clash. It has two consequences, and the second is the
one that cost real work:

1. It OVERWRITES the main fit's ``reason="converged"`` snapshot -- the one
   artefact that makes a fit whose postfit died recoverable. Observed on a fit
   whose fitresults file was ``meta``-only after a postfit crash: the converged
   snapshot was the only record of the minimum, and by the time anyone looked it
   had been replaced.
2. What replaces it has the COMPOSITE layout -- the analysis parameters plus one
   bin scale per projection bin -- so it cannot be loaded back into the main fit
   by ``--externalPostfit`` at all. Meanwhile ``Snapshotter._write`` logs
   "resume with --externalPostfit <path>" for every write, inviting exactly
   that.
"""

import pathlib
import re

from rabbit.snapshot import sibling_snapshot_path

DRIVER = pathlib.Path(__file__).resolve().parents[1] / "bin" / "rabbit_fit.py"


def test_none_stays_none():
    """No ``--snapshotFile`` means no snapshot anywhere, including here."""
    assert sibling_snapshot_path(None, "saturated_ch0") is None


def test_path_differs_from_the_parent_and_keeps_the_suffix():
    parent = "/tmp/out/snapshot_fitresults_MYFIT.hdf5"
    child = sibling_snapshot_path(parent, "saturated_ch0 ptll")
    assert child != parent
    assert child.endswith(".hdf5")
    assert pathlib.PurePath(child).parent == pathlib.PurePath(parent).parent
    assert "snapshot_fitresults_MYFIT" in pathlib.PurePath(child).stem


def test_the_tag_is_sanitised():
    """Callers derive the tag from ``mapping.key``, which carries spaces."""
    child = sibling_snapshot_path("/tmp/s.hdf5", "saturated_Project ch0 ptll")
    assert " " not in child
    assert re.fullmatch(r"[0-9A-Za-z._/-]+", child), child


def test_distinct_mappings_get_distinct_paths():
    """Each projection runs its own saturated fit; they must not collide."""
    a = sibling_snapshot_path("/tmp/s.hdf5", "saturated_ch0 ptll")
    b = sibling_snapshot_path("/tmp/s.hdf5", "saturated_ch0 yll")
    assert a != b


def test_driver_retargets_the_saturated_fitters_snapshot():
    """THE REGRESSION TEST: ``save_hists`` must reassign ``snapshot_file``.

    A source-level check, for the same reason ``test_saturated_blinding.py``
    uses one: the behaviour lives in a script's function, and the deepcopy that
    causes the problem is three lines of driver code with no seam a unit test
    can reach without reimplementing the whole postfit.
    """
    import ast

    src = DRIVER.read_text()
    tree = ast.parse(src)
    body = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "save_hists"
    )
    text = ast.get_source_segment(src, body)

    copy_at = text.find("fitter_saturated = copy.deepcopy(fitter)")
    assign_at = text.find("fitter_saturated.snapshot_file")
    assert copy_at != -1, "save_hists no longer deepcopies the fitter"
    assert assign_at != -1, (
        "save_hists does not retarget fitter_saturated.snapshot_file, so the "
        "saturated fit will snapshot over the main fit's converged snapshot "
        "with a composite-layout vector"
    )
    assert copy_at < assign_at, "retargeted before the deepcopy overwrites it"
