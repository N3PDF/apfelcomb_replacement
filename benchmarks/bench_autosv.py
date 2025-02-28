import shutil

import lhapdf
import numpy as np
import pineappl

from pineko import scale_variations


def benchmark_compute_ren_sv_grid(test_files, tmp_path, test_pdf, lhapdf_path):
    to_test_grid_path = (
        test_files
        / "data"
        / "grids"
        / "400"
        / "ATLAS_TTB_8TEV_LJ_TRAP_totest.pineappl.lz4"
    )
    name_grid = "ATLAS_TTB_8TEV_LJ_TRAP_norensv_fixed.pineappl.lz4"
    grid_path = test_files / "data" / "grids" / "400" / name_grid
    new_grid_path = tmp_path / name_grid
    shutil.copy(grid_path, new_grid_path)
    max_as = 2
    nf = 5
    pdf_name = "NNPDF40_nlo_as_01180"
    already_there_res = scale_variations.compute_ren_sv_grid(
        new_grid_path, max_as - 1, nf
    )
    assert already_there_res == scale_variations.ReturnState.ALREADY_THERE
    order_exist_res = scale_variations.compute_ren_sv_grid(
        new_grid_path, max_as, nf, order_exists=True
    )
    assert order_exist_res == scale_variations.ReturnState.ORDER_EXISTS_FAILURE
    result_state = scale_variations.compute_ren_sv_grid(new_grid_path, max_as, nf)
    assert result_state == scale_variations.ReturnState.SUCCESS
    # We are saving the new grid with the same name of the original
    plusrensv_grid_path = tmp_path / name_grid
    with lhapdf_path(test_pdf):
        pdf = lhapdf.mkPDF(pdf_name)

    to_test_grid = pineappl.grid.Grid.read(to_test_grid_path)
    plusrensv_grid = pineappl.grid.Grid.read(plusrensv_grid_path)
    sv_list = [(0.5, 1.0, 1.0), (2.0, 1.0, 1.0)]  # Only μR have to be tested
    bin_number = to_test_grid.bins()

    to_test_res = to_test_grid.convolve(
        pdg_convs=to_test_grid.convolutions,
        xfxs=[pdf.xfxQ2],
        alphas=pdf.alphasQ2,
        order_mask=np.array([], dtype=bool),
        bin_indices=np.array([], dtype=np.uint64),
        channel_mask=np.array([], dtype=bool),
        xi=sv_list,
    ).reshape(bin_number, len(sv_list))

    plusrensv_res = plusrensv_grid.convolve(
        pdg_convs=plusrensv_grid.convolutions,
        xfxs=[pdf.xfxQ2],
        alphas=pdf.alphasQ2,
        order_mask=np.array([], dtype=bool),
        bin_indices=np.array([], dtype=np.uint64),
        channel_mask=np.array([], dtype=bool),
        xi=sv_list,
    ).reshape(bin_number, len(sv_list))

    rtol = 1.0e-14
    for sv in sv_list:
        for n_res, old_res in zip(
            to_test_res.transpose()[sv_list.index(sv)],
            plusrensv_res.transpose()[sv_list.index(sv)],
        ):
            np.testing.assert_allclose(n_res, old_res, rtol=rtol)
