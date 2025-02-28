import pathlib
import shutil

import lhapdf
from click.testing import CliRunner

from pineko.cli._base import command


def benchmark_check_cli(test_files):
    grid_path = pathlib.Path(
        test_files / "data/grids/400/HERA_NC_225GEV_EP_SIGMARED.pineappl.lz4"
    )
    wrong_grid_path = pathlib.Path(
        test_files / "data/grids/208/HERA_CC_318GEV_EM_SIGMARED.pineappl.lz4"
    )
    eko_path = pathlib.Path(test_files / "data/ekos/400/HERA_NC_225GEV_EP_SIGMARED.tar")
    runner = CliRunner()
    result = runner.invoke(
        command, ["check", "compatibility", str(grid_path), str(eko_path)]
    )
    assert "Success: grids are compatible" in result.output
    wrong_result = runner.invoke(
        command, ["check", "compatibility", str(wrong_grid_path), str(eko_path)]
    )
    assert (
        "Error: Q2 grid in pineappl grid and eko operator are NOT compatible!"
        in wrong_result.output
    )
    wrong_scvar_res = runner.invoke(
        command, ["check", "scvar", str(grid_path), "wrong_string", "2", "0"]
    )
    assert "Invalid value for 'SCALE'" in wrong_scvar_res.output
    ren_res = runner.invoke(
        command, ["check", "scvar", str(grid_path), "ren", "3", "0"]
    )
    assert (
        "Success: grids contain renormalization scale variations for as"
        in ren_res.output
    )
    fact_res = runner.invoke(
        command, ["check", "scvar", str(grid_path), "fact", "3", "0"]
    )
    assert (
        "Success: grids contain factorization scale variations for as"
        in fact_res.output
    )


def benchmark_opcard_cli(tmp_path, test_files):
    grid_path = pathlib.Path(
        test_files / "data/grids/400/HERA_NC_225GEV_EP_SIGMARED.pineappl.lz4"
    )
    default_card_path = pathlib.Path(
        test_files / "data/operator_cards/400/_template.yaml"
    )
    thcard_path = pathlib.Path(test_files / "data" / "theory_cards" / "400.yaml")
    target_path = pathlib.Path(tmp_path / "test_ope_card.yaml")
    runner = CliRunner()
    result = runner.invoke(
        command,
        [
            "opcard",
            str(grid_path),
            str(default_card_path),
            str(thcard_path),
            str(target_path),
        ],
    )
    assert "Success" in result.output


def benchmark_compare_cli(lhapdf_path, test_files, test_pdf):
    grid_path = pathlib.Path(
        test_files / "data/grids/208/LHCB_DY_13TEV_DIMUON.pineappl.lz4"
    )
    fk_path = pathlib.Path(
        test_files / "data/fktables/208/LHCB_DY_13TEV_DIMUON.pineappl.lz4"
    )
    runner = CliRunner()
    with lhapdf_path(test_pdf):
        result = runner.invoke(
            command,
            ["compare", str(fk_path), str(grid_path), "2", "0", "NNPDF40_nlo_as_01180"],
        )
    assert "left" in result.output
    assert "right" in result.output


def benchmark_convolve_cli(test_files, tmp_path):
    grid_path = pathlib.Path(
        test_files / "data/grids/400/HERA_NC_225GEV_EP_SIGMARED.pineappl.lz4"
    )
    eko_path = pathlib.Path(test_files / "data/ekos/400/HERA_NC_225GEV_EP_SIGMARED.tar")
    fk_path = tmp_path / "testfk.pineappl.lz4"
    runner = CliRunner()
    result = runner.invoke(
        command,
        ["convolve", str(fk_path), str(grid_path), "2", "0", str(eko_path)],
    )
    assert "Optimizing for Nf6Ind" in result.output


def benchmark_scaffold_cli(test_empty_proj):
    runner = CliRunner()
    conf_file = test_empty_proj / "pineko.toml"
    # empty project is not correctly configured
    res = runner.invoke(command, ["scaffold", "-c", str(conf_file), "check"])
    assert "Error: Project is not correctly configured." in res.output
    # so we need to create all the folders
    res = runner.invoke(command, ["scaffold", "-c", str(conf_file), "new"])
    # and then I can check again
    res = runner.invoke(command, ["scaffold", "-c", str(conf_file), "check"])
    assert "Success: All the folders are correctly configured" in res.output


def benchmark_gen_sv_cli(test_files, tmp_path):
    runner = CliRunner()
    max_as = "2"
    nf = "5"
    name_grid = "ATLAS_TTB_8TEV_LJ_TRAP_norensv_fixed.pineappl.lz4"
    grid_path = test_files / "data" / "grids" / "400" / name_grid
    new_grid_path = tmp_path / name_grid
    target_path = tmp_path
    shutil.copy(grid_path, new_grid_path)
    res = runner.invoke(
        command,
        ["ren_sv_grid", str(new_grid_path), str(target_path), max_as, nf, "False"],
    )
    assert "ReturnState.SUCCESS" in res.output


def benchmark_kfactor_cli(test_files, tmp_path):
    runner = CliRunner()
    conf_file = test_files / "pineko.toml"
    theory_id = 400
    dataset = "ATLAS_TTB_FAKE"
    kfolder = test_files / "data" / "kfactors"
    order_to_update = "3"
    target_path = tmp_path
    res = runner.invoke(
        command,
        [
            "kfactor",
            "-c",
            str(conf_file),
            str(theory_id),
            str(dataset),
            str(kfolder),
            str(target_path),
            order_to_update,
        ],
    )
    assert "The number of bins match the length of the kfactor" in res.output
