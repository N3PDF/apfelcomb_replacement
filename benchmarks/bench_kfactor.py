import lhapdf
import numpy as np
import pineappl

from pineko import configs, kfactor


def benchmark_kfactor_inclusion(test_files, tmp_path, test_pdf, lhapdf_path):

    base_configs = configs.load(test_files / "pineko.toml")
    configs.configs = configs.defaults(base_configs)

    pto_to_update = 3  # here we want to update NNLO
    pdf_name = "NNPDF40_nnlo_as_01180"
    kfactor.apply_to_dataset(
        400,
        "ATLAS_TTB_FAKE",
        test_files / "data" / "kfactors",
        pto_to_update,
        tmp_path,
    )
    pluskfactor_grid_path = tmp_path / "ATLAS_TTB_8TEV_LJ_TRAP.pineappl.lz4"
    with lhapdf_path(test_pdf):
        pdf = lhapdf.mkPDF(pdf_name)
    pluskfactor_grid = pineappl.grid.Grid.read(pluskfactor_grid_path)
    sv_list = [(1.0, 1.0, 1.0)]  # Only ren sv have to be tested
    bin_number = pluskfactor_grid.bins()
    order_mask_nloQCD = pineappl.boc.Order.create_mask(
        orders=pluskfactor_grid.orders(), max_as=2, max_al=0, logs=True
    )
    order_mask_nnloQCD = pineappl.boc.Order.create_mask(
        orders=pluskfactor_grid.orders(), max_as=3, max_al=0, logs=True
    )
    to_test_res_nlo = pluskfactor_grid.convolve(
        pdg_convs=pluskfactor_grid.convolutions,
        xfxs=[pdf.xfxQ2],
        alphas=pdf.alphasQ2,
        order_mask=order_mask_nloQCD,
        bin_indices=np.array([], dtype=np.uint64),
        channel_mask=np.array([], dtype=bool),
        xi=sv_list,
    ).reshape(bin_number, len(sv_list))
    to_test_res_nnlo = pluskfactor_grid.convolve(
        pdg_convs=pluskfactor_grid.convolutions,
        xfxs=[pdf.xfxQ2],
        alphas=pdf.alphasQ2,
        order_mask=order_mask_nnloQCD,
        bin_indices=np.array([], dtype=np.uint64),
        channel_mask=np.array([], dtype=bool),
        xi=sv_list,
    ).reshape(bin_number, len(sv_list))
    centrals_kfactor, _ = kfactor.read_from_file(
        test_files / "data" / "kfactors" / "CF_QCD_ATLAS_TTB_8TEV_LJ_TRAP.dat"
    )
    rtol = 1.0e-15
    for pred_ratio, kf in zip(
        to_test_res_nnlo.transpose()[0] / to_test_res_nlo.transpose()[0],
        centrals_kfactor,
    ):
        np.testing.assert_allclose(kf, pred_ratio, rtol=rtol)
