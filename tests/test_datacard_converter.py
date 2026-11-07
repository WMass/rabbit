"""Conversion of Combine datacards to the rabbit input format.

The cards are written from scratch into a temporary directory; shape cards get
their templates from a small ROOT file created on the fly with uproot. The
expected yields are the ones Combine uses: the process rate from the datacard
for bins without a 'shapes' line, and the template integral for shape cards.
"""

import os
import pathlib
import subprocess
import sys

import hist
import numpy as np
import pytest

from rabbit.datacard_converter import DatacardConverter

# uproot writes the test ROOT files and is not installed on every runner
uproot = pytest.importorskip("uproot")

REPO = pathlib.Path(__file__).resolve().parents[1]

COUNTING_CARD = """\
bin          SR    CR
observation  160   1500
bin          SR    SR    CR
process      sig   bkg   bkg
process      0     1     1
rate         10    100   1000
"""

WILDCARD_CARD = """\
bin          ch1   ch2
observation  50    60
bin          ch1   ch1   ch2
process      sig   bkg   bkg
process      0     1     1
rate         10    40    50
r_sig rateParam * sig 1.0
r_bkg rateParam ch* bkg 1.0
"""

SHAPES_CARD = """\
shapes * * shapes.root $PROCESS
bin          ch
observation  10
bin          ch   ch
process      sig  bkg
process      0    1
rate         10   10
"""


def convert(tmp_path, text, name="card.txt"):
    card = tmp_path / name
    card.write_text(text)
    return DatacardConverter(str(card)).convert_to_hdf5()


def yields(writer):
    return {
        channel: {process: float(np.sum(norm)) for process, norm in processes.items()}
        for channel, processes in writer.dict_norm.items()
    }


def write_shapes(path, values):
    template = hist.Hist(
        hist.axis.Regular(len(values), 0, len(values)),
        storage=hist.storage.Weight(),
    )
    template.values()[...] = values
    with uproot.recreate(path) as root_file:
        for name in ("sig", "bkg", "data_obs"):
            root_file[name] = template


def test_counting_experiment_uses_process_rates(tmp_path):
    writer = convert(tmp_path, COUNTING_CARD)

    # the observation is the data yield, each process gets its rate from the card
    assert yields(writer) == {
        "SR": {"sig": 10.0, "bkg": 100.0},
        "CR": {"bkg": 1000.0},
    }
    np.testing.assert_allclose(writer.dict_data_obs["SR"], [160.0])
    np.testing.assert_allclose(writer.dict_data_obs["CR"], [1500.0])


def test_rate_param_wildcards_skip_bins_without_process(tmp_path):
    writer = convert(tmp_path, WILDCARD_CARD)

    assert yields(writer) == {
        "ch1": {"sig": 10.0, "bkg": 40.0},
        "ch2": {"bkg": 50.0},
    }
    # 'sig' exists only in ch1, so r_sig must not be added in ch2
    assert list(writer.dict_logkavg["ch1"]["sig"]) == ["r_sig"]
    assert list(writer.dict_logkavg["ch1"]["bkg"]) == ["r_bkg"]
    assert list(writer.dict_logkavg["ch2"]["bkg"]) == ["r_bkg"]
    assert "sig" not in writer.dict_logkavg["ch2"]


@pytest.mark.parametrize(
    "line,match",
    [
        ("r_x rateParam SR* bkg 1.0", "no bin matching 'SR\\*'"),
        ("r_y rateParam * nosuch 1.0", "process matching 'nosuch'"),
    ],
)
def test_rate_param_without_match_raises(tmp_path, line, match):
    with pytest.raises(ValueError, match=match):
        convert(tmp_path, WILDCARD_CARD + line + "\n")


def test_shape_card_uses_histogram_integral(tmp_path):
    write_shapes(tmp_path / "shapes.root", [4.0, 6.0])
    writer = convert(tmp_path, SHAPES_CARD)

    assert yields(writer) == {"ch": {"sig": 10.0, "bkg": 10.0}}
    np.testing.assert_allclose(writer.dict_norm["ch"]["sig"], [4.0, 6.0])


def test_shape_card_rate_mismatch_raises(tmp_path):
    write_shapes(tmp_path / "shapes.root", [4.0, 6.0])

    # the rate line says 20, the bkg template integrates to 10
    with pytest.raises(ValueError, match="Mismatch in normalizations"):
        convert(
            tmp_path,
            SHAPES_CARD.replace("rate         10   10", "rate         10   20"),
        )


def test_shape_card_minus_one_takes_integral(tmp_path):
    write_shapes(tmp_path / "shapes.root", [4.0, 6.0])

    writer = convert(
        tmp_path, SHAPES_CARD.replace("rate         10   10", "rate         -1   -1")
    )

    assert yields(writer) == {"ch": {"sig": 10.0, "bkg": 10.0}}


def test_text2hdf5_cli(tmp_path):
    """The command line entry point converts a counting card end to end."""
    card = tmp_path / "card.txt"
    card.write_text(COUNTING_CARD)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO), env.get("PYTHONPATH")])
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO / "bin" / "rabbit_text2hdf5.py"),
            str(card),
            "--outpath",
            str(tmp_path),
        ],
        check=True,
        env=env,
    )

    from rabbit.inputdata import FitInputData

    input_data = FitInputData(str(tmp_path / "card.hdf5"))
    # normalizations of [sig, bkg] in SR and CR, with SR/CR as the rows
    np.testing.assert_allclose(input_data.norm.numpy(), [[10.0, 100.0], [0.0, 1000.0]])
    np.testing.assert_allclose(input_data.data_obs.numpy(), [160.0, 1500.0])
