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


def _job_block(name, text=None):
    """The lines of the ``name:`` job, delimited by indentation.

    Read as text rather than with PyYAML: the CI runner has no yaml module, and
    an ImportError here is not a skipped test but a collection error that
    aborts the whole `pytest tests/` run -- the exact failure this file exists
    to prevent, which is how it announced itself the first time the job ran.
    The parse is cross-checked against a real YAML parser in
    test_the_text_parse_agrees_with_pyyaml, wherever one is installed.

    Takes the workflow as text so the parser itself can be exercised against
    fixtures below, on the runner as well as here -- the PyYAML cross-check at
    the end of this file skips exactly where the parser is load-bearing.

    Returns None when there is no such job.
    """
    if text is None:
        with open(WORKFLOW) as f:
            text = f.read()
    lines = text.splitlines()
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


# pytest flags that take back part of the directory the check below just
# confirmed. Each re-creates #173 by a different door: a file quietly stops
# running and nothing reports it.
NARROWING_FLAGS = ("--ignore", "--ignore-glob", "--deselect", "-k", "-m")


def _run_commands(block):
    """The shell commands in a job block: ``run:`` values and nothing else.

    A step's ``name:`` is prose, and prose is not a command -- scanning every
    line for the word pytest meant ``name: run pytest over tests just this
    once`` put a bare ``tests`` token in front of the check while the job ran
    one file. Both YAML forms are handled: an inline ``run: cmd`` and a block
    scalar (``run: |``), whose continuation lines are indented under the key.
    """
    cmds, block_indent = [], None
    for line in block:
        stripped = line.strip()
        if block_indent is not None:
            if stripped and (len(line) - len(line.lstrip())) <= block_indent:
                block_indent = None  # the scalar ended; re-read this line
            else:
                if stripped:
                    cmds.append(stripped)
                continue
        m = re.match(r"^(\s*)(?:-\s+)?run:\s*(.*)$", line)
        if not m:
            continue
        # a YAML comment needs whitespace before the '#', so this does not
        # truncate a command that contains one
        value = re.split(r"\s#", m.group(2), maxsplit=1)[0].strip()
        if value in ("|", ">", "|-", ">-", "|+", ">+"):
            block_indent = len(line) - len(line.lstrip())
        elif value:
            cmds.append(value)
    return cmds


def _pytest_args(words):
    """The arguments pytest itself receives, i.e. those after the pytest token.

    ``python -m pytest tests/`` puts a ``-m`` in front of pytest that belongs
    to python; reading it as pytest's mark selector would flag the repo's own
    command as narrowing.
    """
    for i, word in enumerate(words):
        if word == "pytest" or word.endswith("/pytest"):
            return words[i + 1 :]
    return []


def _narrowing_flags(words):
    args = _pytest_args(words)
    return [
        a
        for a in args
        for flag in NARROWING_FLAGS
        if a == flag or a.startswith(flag + "=")
    ]


def _pytest_commands(job, text=None):
    """Shell words of each command in ``job`` that invokes pytest."""
    block = _job_block(job, text)
    if block is None:
        return None
    cmds = []
    for cmd in _run_commands(block):
        if "pytest" not in cmd:
            continue
        try:
            cmds.append(shlex.split(cmd))
        except ValueError:
            continue
    return cmds


def _job_names(text=None):
    """Top-level job names, read as text (see _job_block)."""
    if text is None:
        with open(WORKFLOW) as f:
            text = f.read()
    lines = text.splitlines()
    names, indent = [], None
    for i, line in enumerate(lines):
        if re.match(r"^jobs:\s*$", line):
            indent = None
            for later in lines[i + 1 :]:
                if not later.strip() or later.lstrip().startswith("#"):
                    continue
                if indent is None:
                    indent = len(later) - len(later.lstrip())
                depth = len(later) - len(later.lstrip())
                if depth < indent:
                    break
                m = re.match(rf"^\s{{{indent}}}([A-Za-z0-9_-]+):\s*$", later)
                if m and depth == indent:
                    names.append(m.group(1))
            break
    return names


def _needs(block):
    """The jobs a block declares in ``needs:``, in any of YAML's three forms."""
    out = []
    for i, line in enumerate(block):
        m = re.match(r"^\s*needs:\s*(.*)$", line)
        if not m:
            continue
        value = re.split(r"\s#", m.group(1), maxsplit=1)[0].strip()
        if value.startswith("["):
            out += [
                v.strip().strip("'\"")
                for v in value.strip("[]").split(",")
                if v.strip()
            ]
        elif value:
            out.append(value.strip("'\""))
        else:  # a block list on the following lines
            for later in block[i + 1 :]:
                item = re.match(r"^\s*-\s*(\S+)\s*$", later)
                if not item:
                    break
                out.append(item.group(1).strip("'\""))
    return out


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
    assert any({"tests", "tests/"} & set(_pytest_args(words)) for words in cmds), (
        f"the `{JOB}` job no longer runs pytest over the whole tests/ "
        f"directory, so it does not cover every file; it runs: {cmds!r}"
    )
    narrowed = [f for words in cmds for f in _narrowing_flags(words)]
    assert not narrowed, (
        f"the `{JOB}` job runs pytest over tests/ and then narrows it back "
        f"down with {narrowed}. That is #173 by a different door: a file stops "
        "running and nothing reports it, which is what collecting the whole "
        "directory exists to prevent. Quarantine the case in the test file "
        "itself -- xfail or skip with a reason -- so the omission stays "
        "visible where it happens."
    )


# Workflow shapes the text parser has to get right, exercised directly rather
# than through main.yml. The PyYAML cross-check at the end of this file is the
# obvious way to keep a hand-rolled parse honest, but it skips on the CI runner
# -- which has no yaml module, the very reason the parse is hand-rolled -- so
# it is inert exactly where the parser is load-bearing. These run everywhere.
_FIXTURES = {
    "inline": """
jobs:
  all-unit-tests:
    steps:
      - name: run the whole test suite
        run: python -m pytest tests/ -q
  other:
    needs: all-unit-tests
""",
    "block scalar": """
jobs:
  all-unit-tests:
    steps:
      - name: run the whole test suite
        run: |
          export PYTHONPATH=.
          python -m pytest tests/ -q
  other:
    needs: [all-unit-tests]
""",
    # the hole this parser had: prose that mentions pytest and tests is not a
    # command, but was read as one, and its bare `tests` token satisfied the
    # coverage check while the job ran a single file
    "prose name": """
jobs:
  all-unit-tests:
    steps:
      - name: run pytest over tests just this once
        run: pytest tests/test_bbstat.py
""",
    # covers the directory and then takes most of it back
    "narrowed": """
jobs:
  all-unit-tests:
    steps:
      - name: run the whole test suite
        run: pytest tests/ --ignore=tests/test_native_minimizer.py -k "not restart"
""",
}


@pytest.mark.parametrize("label", ["inline", "block scalar"])
def test_the_parser_reads_a_real_command(label):
    assert _pytest_commands(JOB, text=_FIXTURES[label]) == [
        ["python", "-m", "pytest", "tests/", "-q"]
    ]


@pytest.mark.parametrize("label", ["prose name", "narrowed"])
def test_a_job_that_does_not_cover_the_directory_is_caught(label):
    """The guard's own failure modes, checked against the parser it uses."""
    cmds = _pytest_commands(JOB, text=_FIXTURES[label])
    covers = any({"tests", "tests/"} & set(_pytest_args(w)) for w in cmds)
    narrowed = [f for w in cmds for f in _narrowing_flags(w)]
    assert not (covers and not narrowed), (
        f"the {label!r} workflow would satisfy the coverage check while "
        f"running less than tests/: {cmds!r}"
    )


def test_python_dash_m_is_not_read_as_a_mark_selector():
    """`python -m pytest` must not read as `pytest -m`, or the repo's own
    command counts as narrowing and the check fires on every PR."""
    assert _narrowing_flags(["python", "-m", "pytest", "tests/", "-q"]) == []
    assert _narrowing_flags(["pytest", "tests/", "-m", "slow"]) == ["-m"]


def test_no_job_depends_on_a_job_that_does_not_exist():
    """The matrix job was removed; seven jobs had named it in `needs`.

    GitHub rejects the whole workflow for a dangling dependency, so this would
    surface as every job failing to start rather than as a test failure. Read
    as text, like everything else here, so it runs on the runner too.
    """
    names = _job_names()
    assert JOB in names, f"the text parse found no `{JOB}` job: {names}"
    dangling = [
        f"{name} -> {n}"
        for name in names
        for n in _needs(_job_block(name))
        if n not in names
    ]
    assert not dangling, f"jobs depend on jobs that do not exist: {dangling}"


def test_the_text_parse_agrees_with_pyyaml():
    """Wherever PyYAML is installed, hold the hand-rolled parse to it.

    Supplementary now that the fixtures above pin the parser without it: this
    catches a drift between the text parse and what the workflow *means*,
    rather than between the parse and the shapes anticipated for it.
    """
    yaml = pytest.importorskip("yaml", reason="no PyYAML here; fixtures cover it")
    with open(WORKFLOW) as f:
        jobs = yaml.safe_load(f)["jobs"]

    assert (JOB in jobs) == (
        _pytest_commands(JOB) is not None
    ), "the text parse disagrees with PyYAML on whether the unit test job exists"
    assert sorted(_job_names()) == sorted(jobs), (
        "the text parse and PyYAML disagree on the job names: "
        f"{sorted(_job_names())} vs {sorted(jobs)}"
    )

    steps = " ".join(s.get("run", "") for s in jobs[JOB].get("steps", []))
    assert ("pytest" in steps) == bool(
        _pytest_commands(JOB)
    ), "the text parse disagrees with PyYAML on whether the job runs pytest"

    for name, job in jobs.items():
        needs = job.get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        assert sorted(_needs(_job_block(name))) == sorted(
            needs
        ), f"the text parse and PyYAML disagree on `needs` for {name}"


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
