"""Tools to compare grids and FK tables."""

import numpy as np
import pandas as pd
import pineappl


def compare(pine, fktable, max_as, max_al, pdfs, scales):
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
    pdfs : list(str)
        list of the PDF set names
    scales: tuple
        contains the values the renormalization, factorization, and fragmentation scale
        variations

    Returns
    -------
    df : pd.DataFrame
        comparison table
    """
    import lhapdf  # pylint: disable=import-error,import-outside-toplevel

    pdfsets = [lhapdf.mkPDF(pdf, 0) for pdf in pdfs]

    # Check compatibilty between the FK table and the grid
    assert len(fktable.convolutions) == len(
        pine.convolutions
    ), "FK table and Grid have different (number of) convolutions"

    for gconv, fconv in zip(pine.convolutions, fktable.convolutions):
        if gconv.convolution_types.polarized != fconv.convolution_types.polarized:
            raise ValueError(
                "The Grid and FK table do not have the same type of Polarization:"
                f"grid={gconv.convolution_types.polarized}"
                f" vs. fk={fconv.convolution_types.polarized}"
            )
        if gconv.convolution_types.time_like != fconv.convolution_types.time_like:
            raise ValueError(
                "The Grid and FK table do not have the same type of Interval:"
                f"grid={gconv.convolution_types.time_like}"
                f" vs. fk={fconv.convolution_types.time_like}"
            )

    # TODO: Add checks that verify the compatibility with the PDFs
    order_mask = pineappl.boc.Order.create_mask(
        orders=pine.orders(), max_as=max_as, max_al=max_al, logs=True
    )
    # Perform the convolutions of the Grids and FK tables (This is now much simpler!)
    pine_predictions = pine.convolve(
        pdg_convs=pine.convolutions,
        xfxs=[pdf.xfxQ2 for pdf in pdfsets],
        alphas=pdfsets[0].alphasQ2,  # TODO: Choose which PDF should be used
        order_mask=order_mask,
        xi=[scales],  # TODO: Exposes as a list
    )
    fktable_predictions = fktable.convolve(
        pdg_convs=fktable.convolutions,
        xfxs=[pdf.xfxQ2 for pdf in pdfsets],
    )
    before = np.array(pine_predictions)
    after = np.array(fktable_predictions)

    df = pd.DataFrame()
    # add bin info
    bin_specs = np.array(pine.bin_limits())
    for d in range(pine.bin_dimensions()):
        df[f"O{d+1} left"] = bin_specs[:, d, 0]
        df[f"O{d+1} right"] = bin_specs[:, d, 1]
    # add data
    df["PineAPPL"] = before
    df["FkTable"] = after
    df["permille_error"] = (after / before - 1.0) * 1000.0
    return df
