"""
Suite of tests that go through the entire process of creating a new fktable
from a empty folder.

The target theory is 400 and the relevant `.toml`, theory runcard and eko template
are downloaded from https://github.com/NNPDF/theories during this test so this tests
has the double effect of ensuring compatibility between both repositories.
"""

import itertools
from pathlib import Path
from subprocess import run
from urllib.request import urlretrieve

import numpy as np
import pytest
from eko.interpolation import XGrid
from eko.io.runcards import OperatorCard
from pineappl.fk_table import FkTable
from yaml import dump, safe_load

THEORIES_REPO = "https://raw.githubusercontent.com/NNPDF/theories/main"
THEORYID = 400
REGRESSION_ROOT = Path(__file__).parent / "regression_data"


def _download_resources(filename, tmp_path):
    """Download resources (filename) from the theories repo and put it in the same path
    relative to THEORIES_REPO
    """
    output_file = tmp_path / filename
    output_file.parent.mkdir(exist_ok=True)
    urlretrieve(f"{THEORIES_REPO}/{filename}", output_file)
    return output_file


def _download_dataset(dataset, theoryid, tmp_path):
    """Download both the yaml file and all grids for a given dataset for a given theory"""
    yaml_file = f"data/yamldb/{theoryid}/{dataset}.yaml"
    _download_resources(yaml_file, tmp_path)
    # However, the yaml file goes into the "root" of the yaml directory, move it there!
    right_yaml = tmp_path / "data" / "yamldb" / f"{dataset}.yaml"
    (tmp_path / yaml_file).rename(right_yaml)
    # Download the relevant grids for this dataset
    res = safe_load(right_yaml.read_text(encoding="utf-8"))
    grids = list(itertools.chain(*res["operands"]))
    for grid_name in grids:
        _download_resources(f"data/grids/400/{grid_name}.pineappl.lz4", tmp_path)
    return grids


class _FakePDF:
    """A Fake lhapdf-like PDF to evolve the grids"""

    def __init__(self):
        pids = np.arange(-6, 8)
        pids[6] = 21
        pids[-1] = 22

        alphas = np.linspace(1.2, 1.8, len(pids))
        betas = np.linspace(1.2, 3.8, len(pids))

        self._alphas = dict(zip(pids, alphas))
        self._betas = dict(zip(pids, betas))

    def xfxQ2(self, pid, x, q2):
        """Compute x^alpha*(1-x)^beta"""
        alpha = self._alphas[pid]
        beta = self._betas[pid]
        return np.power(x, alpha) * np.power(1 - x, beta)


def _trim_template(template_card, take_points=10):
    """Trim the template card so that the number of x-values to compute is much smaller"""
    card_info = OperatorCard.from_dict(
        safe_load(template_card.read_text(encoding="utf-8"))
    )
    original_x = card_info.xgrid
    size = len(original_x.raw)
    skip = int(size / take_points)
    card_info.xgrid = XGrid(original_x.raw[:size:skip])
    template_card.write_text(dump(card_info.raw), encoding="utf-8")


@pytest.mark.parametrize("dataset", ["LHCBWZMU8TEV", "INTEGXT3"])
def benchmark_regression(tmp_path, dataset):
    """Run pineko through subprocess to ensure that the shell scripts are working exactly
    as intended.

    If the data does not exist, create it and fail the test, i.e., in order to regenerate
    the data just remove the previous dataset.npy file
    """
    # We start by downloading pineko.toml in order to generate the folder structure
    _download_resources("pineko.toml", tmp_path)
    # Which we create... now!
    run(["pineko", "scaffold", "new"], cwd=tmp_path, check=True)

    # Now download other necessary objects
    _download_resources(f"data/theory_cards/{THEORYID}.yaml", tmp_path)
    template_card = _download_resources(
        f"data/operator_cards/{THEORYID}/_template.yaml", tmp_path
    )
    _trim_template(template_card)

    # And use some small (but not trivial!) dataset to test
    gridnames = _download_dataset(dataset, THEORYID, tmp_path)

    # Now go, first with eko creation
    run(
        ["pineko", "theory", "opcards", str(THEORYID), dataset],
        cwd=tmp_path,
        check=True,
    )
    run(["pineko", "theory", "ekos", str(THEORYID), dataset], cwd=tmp_path, check=True)

    # Then FK Table production!
    run(["pineko", "theory", "fks", str(THEORYID), dataset], cwd=tmp_path, check=True)

    # Now loop over the grids and check the results of the convolution with the PDF
    pdf = _FakePDF()
    regression_path = REGRESSION_ROOT / f"{dataset}.npy"

    result = []
    for grid_name in gridnames:
        fkt = FkTable.read(
            tmp_path / "data" / "fktables" / str(THEORYID) / f"{grid_name}.pineappl.lz4"
        )
        result.append(fkt.convolve(pdg_convs=fkt.convolutions, xfxs=[pdf.xfxQ2]))
    result = np.concatenate(result)

    if not regression_path.exists():
        np.save(regression_path, result)
        raise FileNotFoundError("Regression did not exist and has been regenerated")

    regression_data = np.load(regression_path)
    np.testing.assert_allclose(regression_data, result, rtol=4e-6)
