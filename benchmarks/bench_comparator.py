import numpy as np
import pineappl

import pineko


def benchmark_compare(lhapdf_path, test_files, test_pdf):
    pine_path = test_files / "data/grids/208/LHCB_DY_13TEV_DIMUON.pineappl.lz4"
    grid = pineappl.grid.Grid.read(pine_path)
    fk_path = test_files / "data/fktables/208/LHCB_DY_13TEV_DIMUON.pineappl.lz4"
    fk = pineappl.fk_table.FkTable.read(fk_path)
    pdfs = ["NNPDF40_nlo_as_01180"]
    xi = (1.0, 1.0, 1.0)
    with lhapdf_path(test_pdf):
        comp_table = pineko.comparator.compare(grid, fk, 2, 0, pdfs, xi)
    errors = comp_table["permille_error"].values
    assert np.all(errors < 5.0)
