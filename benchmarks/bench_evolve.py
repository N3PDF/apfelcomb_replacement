import json
import pathlib

import eko
import eko.io.legacy
import numpy as np
import pineappl
import pytest
import yaml

import pineko
import pineko.evolve
import pineko.theory_card


@pytest.mark.parametrize("theoryid,is_mixed", [(400, False), (208, True)])
def benchmark_write_operator_card_from_file_num_fonll(
    tmp_path, test_files, test_configs, theoryid, is_mixed
):
    tcard = pineko.theory_card.load(theoryid)
    tcards_path_list = pineko.fonll.dump_tcards(tcard, tmp_path, theoryid)
    pine_path = (
        test_files
        / "data"
        / "grids"
        / "400"
        / "HERA_NC_225GEV_EP_SIGMARED.pineappl.lz4"
    )
    default_path = test_files / "data" / "operator_cards" / "400" / "_template.yaml"
    num_opcard = 7 if is_mixed else 5
    targets_path_list = [
        tmp_path / f"test_opcard_{num}.yaml" for num in range(num_opcard)
    ]
    for target_path, tcard_path in zip(targets_path_list, tcards_path_list):
        with open(tcard_path, encoding="utf-8") as f:
            tcard = yaml.safe_load(f)
        _x_grid, _q2_grid = pineko.evolve.write_operator_card_from_file(
            pine_path, default_path, target_path, tcard
        )
    # Check if the opcards are ok
    for opcard_path, cfg in zip(
        targets_path_list,
        pineko.fonll.FNS_CONFIG,
    ):
        with open(opcard_path, encoding="utf-8") as f:
            ocard = yaml.safe_load(f)
        for entry in ocard["mugrid"]:
            assert entry[1] == int(cfg.nf)


def benchmark_write_operator_card_from_file(tmp_path, test_files, test_configs):
    pine_path = test_files / "data/grids/400/HERA_NC_225GEV_EP_SIGMARED.pineappl.lz4"
    default_path = test_files / "data/operator_cards/400/_template.yaml"
    target_path = pathlib.Path(tmp_path / "test_operator.yaml")
    tcard = pineko.theory_card.load(400)
    x_grid, _q2_grid = pineko.evolve.write_operator_card_from_file(
        pine_path, default_path, target_path, tcard
    )

    # Load the operator card
    myopcard = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    # Check if it contains all the information for eko
    assert np.allclose(myopcard["xgrid"], x_grid)

    wrong_pine_path = test_files / "data/grids/208/HERA_CC_318GEV_EM_wrong.pineappl.lz4"
    with pytest.raises(FileNotFoundError):
        _ = pineko.evolve.write_operator_card_from_file(
            wrong_pine_path, default_path, target_path, 1.0
        )


def benchmark_dglap(tmp_path, test_files, test_configs):
    pine_path = test_files / "data/grids/400/HERA_NC_225GEV_EP_SIGMARED.pineappl.lz4"
    default_path = test_files / "data/operator_cards/400/_template.yaml"
    target_path = pathlib.Path(tmp_path / "test_operator.yaml")

    theory_id = 400
    tcard = pineko.theory_card.load(theory_id)
    # In order to check if the operator card is enough for eko, let's compute the eko

    pineko.evolve.write_operator_card_from_file(
        pine_path, default_path, target_path, tcard
    )

    # Load the opcard
    myopcard = yaml.safe_load(target_path.read_text(encoding="utf-8"))

    # I need smaller x and q grids in order to compute a small eko
    small_x_grid = np.geomspace(1e-3, 1.0, 5)
    target = (10.0, 5)
    myopcard["xgrid"] = small_x_grid
    myopcard["mugrid"] = [target]
    legacy_class = eko.io.runcards.Legacy(tcard, myopcard)
    new_theory = legacy_class.new_theory
    new_op = eko.io.runcards.OperatorCard.from_dict(myopcard)

    eko_path = pathlib.Path(tmp_path / "test_eko.tar")
    eko.runner.solve(new_theory, new_op, eko_path)


def benchmark_evolve_grid(tmp_path, lhapdf_path, test_files, test_pdf):
    pine_path = test_files / "data/grids/400/HERA_NC_225GEV_EP_SIGMARED.pineappl.lz4"
    pinegrid = pineappl.grid.Grid.read(pine_path)
    eko_path = test_files / "data/ekos/400/HERA_NC_225GEV_EP_SIGMARED.tar"
    target_path = pathlib.Path(tmp_path / "test_fktable.pineappl.lz4")
    max_as = 3
    max_al = 0
    base_configs = pineko.configs.load(test_files)
    pineko.configs.configs = pineko.configs.defaults(base_configs)
    tcard = pineko.theory_card.load(400)
    assumptions = pineko.theory_card.construct_assumptions(tcard)
    with eko.EKO.edit(eko_path) as eko_op:
        with lhapdf_path(test_pdf):
            pineko.evolve.evolve_grid(
                grid=pinegrid,
                operators=[eko_op],
                fktable_path=target_path,
                max_as=max_as,
                max_al=max_al,
                xir=1.0,
                xif=1.0,
                xia=1.0,
                assumptions=assumptions,
                comparison_pdfs=["NNPDF40_nnlo_as_01180"],
            )
            # check metadata is there - fixes https://github.com/NNPDF/pineko/issues/70
            fk = pineappl.fk_table.FkTable.read(target_path)
            kvs = fk.metadata
            assert "results_fk" in kvs
            assert "eko_theory_card" in kvs
            assert json.dumps(eko_op.theory_card.raw) == kvs["eko_theory_card"]
