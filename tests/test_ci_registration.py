"""Every test file must be reachable by CI.

CI runs the unit tests one way: an `all-unit-tests` job invoking
`pytest tests/`, which discovers test files rather than being told about them.
That is the whole point. It replaced a hand-maintained matrix of filenames in
which an unlisted file ran nowhere and nothing reported the omission -- the
default was "don't run", a PR adding tests went green either way, and 137 tests
across eight files had accumulated unrun before anyone measured it (issue #173).

Discovery closes that by construction, but it moves the failure rather than
removing it: a file pytest cannot collect anything from is now the silent case,
since there is no second mechanism to catch it. The four script-style files
that used to need the matrix (a `main()` rather than `test_` functions) were
converted for exactly that reason. These tests keep it that way.

What is deliberately not guarded: deleting the `all-unit-tests` job itself.
Nothing would then run this file either. That is a visible, reviewable act --
every unit-test check disappears from the PR at once -- rather than the quiet
drift this module exists to catch, and a tripwire job would mean running some
test file twice, which is what the unification removed.
"""

import ast
import os
import re
import shlex

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
WORKFLOW = os.path.join(REPO, ".github", "workflows", "main.yml")
JOB = "all-unit-tests"

# What `pytest tests/` picks up, by pytest's default naming conventions.
PYTEST_FUNC_PREFIX = "test"
PYTEST_CLASS_PREFIX = "Test"


def _job_block(name):
    """The lines of the ``name:`` job, delimited by indentation.

    Read as text rather than with PyYAML: the CI runner has no yaml module, and
    an ImportError here is not a skipped test but a collection error that
    aborts the whole `pytest tests/` run -- the exact failure this file exists
    to prevent, which is how it announced itself the first time the job ran.
    The parse is cross-checked against a real YAML parser in
    test_the_text_parse_agrees_with_pyyaml, wherever one is installed.

    Returns None when there is no such job.
    """
    with open(WORKFLOW) as f:
        lines = f.read().splitlines()
    out, indent = None, None
    for line in lines:
        if out is None:
            if re.match(rf"^(\s*){re.escape(name)}:\s*$", line):
                indent = len(line) - len(line.lstrip())
                out = []
            continue
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return out


def _test_files():
    names = sorted(
        n for n in os.listdir(TESTS) if n.startswith("test_") and n.endswith(".py")
    )
    assert names, "no test files found; the test is looking in the wrong place"
    return names


def _pytest_commands(job):
    """Shell words of each command in ``job`` that invokes pytest.

    Scans every line of the block rather than only inline ``run:`` values, so a
    command inside a ``run: |`` block scalar counts too.
    """
    block = _job_block(job)
    if block is None:
        return None
    cmds = []
    for line in block:
        text = re.sub(r"^-?\s*run:\s*", "", line.split("#", 1)[0].strip())
        if "pytest" in text:
            try:
                cmds.append(shlex.split(text))
            except ValueError:
                continue
    return cmds


def _collects_under_pytest(name):
    """Whether `pytest tests/<name>` would collect at least one test.

    Static, by pytest's default conventions (top-level ``test_*`` functions and
    ``Test*`` classes), so this test stays hermetic -- it does not import the
    module or shell out to a nested pytest run.
    """
    with open(os.path.join(TESTS, name)) as f:
        tree = ast.parse(f.read(), filename=name)
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith(PYTEST_FUNC_PREFIX):
            return True
        if isinstance(node, ast.ClassDef) and node.name.startswith(PYTEST_CLASS_PREFIX):
            return True
    return False


@pytest.mark.parametrize("name", _test_files())
def test_every_test_file_is_collected_by_pytest(name):
    """A file pytest collects nothing from runs nowhere, silently."""
    assert _collects_under_pytest(name), (
        f"tests/{name} defines no top-level test_* function or Test* class, so "
        "`pytest tests/` collects nothing from it and it runs nowhere in CI. "
        "Script-style tests (a main() that asserts) are not picked up -- give "
        "it test_* functions, as the four converted files do."
    )


def test_the_unit_test_job_runs_the_whole_suite():
    """Everything above assumes `pytest tests/` actually runs, over tests/."""
    cmds = _pytest_commands(JOB)
    assert (
        cmds is not None
    ), f"the `{JOB}` job is gone; no unit tests run in CI at all (see #173)"
    # Token-wise, not substring: `pytest tests/test_bbstat.py` contains the
    # string "pytest tests/" while covering one file out of twenty.
    assert any({"tests", "tests/"} & set(words) for words in cmds), (
        f"the `{JOB}` job no longer runs pytest over the whole tests/ "
        f"directory, so it does not cover every file; it runs: {cmds!r}"
    )


def test_the_text_parse_agrees_with_pyyaml():
    """Wherever PyYAML is installed, hold the hand-rolled parse to it.

    The CI runner has no yaml module, so the parser above cannot use one; this
    keeps it honest anywhere that does, rather than letting a text parse drift
    from what the workflow actually means.
    """
    yaml = pytest.importorskip("yaml", reason="no PyYAML here; parser runs unchecked")
    with open(WORKFLOW) as f:
        jobs = yaml.safe_load(f)["jobs"]

    assert (JOB in jobs) == (
        _pytest_commands(JOB) is not None
    ), "the text parse disagrees with PyYAML on whether the unit test job exists"

    steps = " ".join(s.get("run", "") for s in jobs[JOB].get("steps", []))
    assert ("pytest" in steps) == bool(
        _pytest_commands(JOB)
    ), "the text parse disagrees with PyYAML on whether the job runs pytest"


def test_no_job_depends_on_a_job_that_does_not_exist():
    """The matrix job was removed; seven jobs had named it in `needs`.

    GitHub rejects the whole workflow for a dangling dependency, so this would
    surface as every job failing to start rather than as a test failure.
    """
    yaml = pytest.importorskip("yaml", reason="no PyYAML here")
    with open(WORKFLOW) as f:
        jobs = yaml.safe_load(f)["jobs"]
    dangling = []
    for name, job in jobs.items():
        needs = job.get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        dangling += [f"{name} -> {n}" for n in needs if n not in jobs]
    assert not dangling, f"jobs depend on jobs that do not exist: {dangling}"


def test_cross_test_imports_are_not_bare():
    """A bare `from test_x import ...` breaks the suite at collection time.

    It resolves only when tests/ is itself on sys.path, which `pytest tests/`
    does not arrange, and the resulting ImportError aborts collection for the
    *whole run*, not just that file -- which is how one stray import kept the
    directory-wide job from being turned on at all (issue #173).

    Both the forms already used in this suite are fine: relative
    (`from .test_x import ...`) and package-qualified
    (`from tests.test_x import ...`). Only the bare absolute form is flagged.
    """
    offenders = []
    for name in _test_files():
        with open(os.path.join(TESTS, name)) as f:
            tree = ast.parse(f.read(), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import (`from .test_x import ...`),
                # which resolves through the tests package and is fine.
                if node.level == 0 and (node.module or "").startswith("test_"):
                    offenders.append(f"tests/{name}:{node.lineno}: {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("test_"):
                        offenders.append(f"tests/{name}:{node.lineno}: {alias.name}")
    assert not offenders, (
        "bare cross-test imports break `pytest tests/` at collection time; "
        "use `tests.<module>` or a relative `.<module>`:\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
