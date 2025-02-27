"""Tools to compare grids and FK tables."""

import numpy as np
import pandas as pd
import pineappl
import rich


class GridtoFKError(Exception):
    """Raised when the difference between the Grid and FK table is above some threshold."""


def compare(pine, fktable, max_as, max_al, pdf1, xir, xif, threshold=5.0, pdf2=None):
    """Build comparison table.

    Parameters
    ----------
    pine : pineappl.grid.Grid
        uncovoluted grid
    fktable : pineappl.fktable.FkTable
        convolved grid
    max_as : int
        maximum power of strong coupling
    max_al : int
        maximum power of electro-weak coupling
    pdf1 : str
        PDF set name
    xir : float
        renormalization scale variation
    xif : float
        factorization scale variation
    threshold: float
        check if the difference between the Grid and FK table is above the
        threshold then raise an error
    pdf2: str or None
        PDF set for the second convolution, if different from the first

    Returns
    -------
    df : pd.DataFrame
        comparison table
    """
    import lhapdf  # pylint: disable=import-error,import-outside-toplevel

    pdfset1 = lhapdf.mkPDF(pdf1, 0)
    if pdf2 is not None:
        pdfset2 = lhapdf.mkPDF(pdf2, 0)
    else:
        pdfset2 = pdfset1

    # TODO: This should probably changed in the future to use the Grid::convolutions
    try:
        parton1 = int(pine.key_values()["convolution_particle_1"])
        parton2 = int(pine.key_values()["convolution_particle_2"])

        if (
            fktable.key_values()["convolution_type_1"]
            != pine.key_values()["convolution_type_1"]
            or fktable.key_values()["convolution_type_2"]
            != pine.key_values()["convolution_type_2"]
        ):
            raise ValueError(
                f"""Grids and FkTables do not have the same convolution types:
                grid=({pine.key_values()['convolution_type_1']},{pine.key_values()['convolution_type_2']})
                vs. fk=({fktable.key_values()['convolution_type_1']},{fktable.key_values()['convolution_type_2']})"""
            )

        # log some useful info to check if PDFs are swapped
        rich.print(
            f"[yellow]Convolution type 1: {fktable.key_values()['convolution_type_1']}, PDF 1: {pdf1}"
        )
        rich.print(
            f"[yellow]Convolution type 2: {fktable.key_values()['convolution_type_2']}, PDF 2: {pdf2}"
        )

    except KeyError:
        parton1 = int(pine.key_values()["initial_state_1"])
        parton2 = int(pine.key_values()["initial_state_2"])
    hadronic = parton1 == parton2

    order_mask = pineappl.grid.Order.create_mask(pine.orders(), max_as, max_al, True)
    if hadronic:
        before = np.array(
            pine.convolve_with_two(
                pdg_id1=parton1,
                xfx1=pdfset1.xfxQ2,
                pdg_id2=parton2,
                xfx2=pdfset2.xfxQ2,
                alphas=pdfset1.alphasQ2,
                order_mask=order_mask,
                xi=((xir, xif),),
            )
        )
        after = np.array(
            fktable.convolve_with_two(parton1, pdfset1.xfxQ2, parton2, pdfset2.xfxQ2)
        )
    else:
        before = np.array(
            pine.convolve_with_one(
                parton1,
                pdfset1.xfxQ2,
                pdfset1.alphasQ2,
                order_mask=order_mask,
                xi=((xir, xif),),
            )
        )
        after = np.array(fktable.convolve_with_one(parton1, pdfset1.xfxQ2))

    df = pd.DataFrame()
    # add bin info
    for d in range(pine.bin_dimensions()):
        try:
            label = pine.key_values()[f"x{d+1}_label"]
        except KeyError:
            label = f"O{d+1}"
        df[f"{label} left"] = pine.bin_left(d)
        df[f"{label} right"] = pine.bin_right(d)
    # add data
    df["PineAPPL"] = before
    df["FkTable"] = after
    df["permille_error"] = (after / before - 1.0) * 1000.0

    if (df["permille_error"].abs() >= threshold).any():
        raise GridtoFKError(
            f"The difference between the Grid and FK is above {threshold} permille."
        )

    return df
