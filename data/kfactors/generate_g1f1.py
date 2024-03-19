import lhapdf
import pineappl
from glob import glob
from datetime import datetime as dt
import numpy as np
import os
import argparse
from typing import List, Tuple


def get_file(folder: str, data: str, theory: str) -> Tuple[str, List[str]]:
    """
    Get a list of paths to PineAPPL grids in the specified folder.

    Parameters
    ----------
    folder : str
        The folder path where PineAPPL grids are located.
    data : str
        Name of the commondata set.

    Returns
    -------
    pdf_name : str
        The name of the PDF dataset.
    gpaths : List[str]
        List of paths to PineAPPL grid files.
    """
    file = glob(folder + f"/{data}_F1.pineappl.lz4")[0]
    return file


def get_prediction(gpath: str, pdf_name: str) -> np.ndarray:
    """
    Get predictions by convoluting a PineAPPL grid with a LHAPDF PDF.

    Parameters
    ----------
    gpath : str
        Path to the PineAPPL grid file.
    pdf_name : str
        The name of the LHAPDF dataset.

    Returns
    -------
    prediction : np.ndarray
        Computed predictions.
    """
    # Load the PineAPPL grid
    grid = pineappl.grid.Grid.read(gpath)

    # Load the LHAPDF
    pdf = lhapdf.mkPDF(pdf_name)

    # Proton reference number
    nr = 2212

    # Perform the convolution
    prediction = grid.convolute_with_one(
        nr,  # Type of target
        pdf.xfxQ2,  # The PDF callable pdf.xfxQ2(pid, x, Q2) -> xfx
        pdf.alphasQ2,  # The alpha_s callable pdf.alpha_s(Q2) -> alpha_s
    )

    # Compute the k-factor (1 / F1)
    prediction = 1 / prediction

    return prediction


def save_data(
    data: np.ndarray,
    dataset_name: str,
    pdf_name: str,
    author_name: str,
    theory_name: str,
    output_name: str = "results",
):
    """
    Save computed data to a file with metadata.

    Parameters
    ----------
    data : np.ndarray
        Computed data.
    dataset_name : str
        Name of the dataset.
    author_name : str
        Name of the author.
    theory_name : str
        Name of the theory.
    output_name : str, optional
        Output folder name, default is "results".
    """
    strf_data = ""
    for i in range(data.shape[0]):
        strf_data += f"{data[i]}  0.0\n"

    date = dt.now().date()
    string = (
        f"""********************************************************************************
SetName: {dataset_name}
Author: {author_name}
Date: {date}
CodesUsed: https://github.com/NNPDF/yadism
TheoryInput: {theory_name}
PDFset: {pdf_name}
Warnings: F1 normalization for {dataset_name}
********************************************************************************
"""
        + strf_data
    )
    dataset_name += "_G1"
    with open(output_name + f"/CF_NRM_{dataset_name}.dat", "w") as file:
        file.write(string)


# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument("pdf", help="The name of the PDF dataset of LHAPDF")
parser.add_argument("folder", help="The folder name of the F1 pineapple grids")
parser.add_argument("data", help="Name of the commondata set")
parser.add_argument("--author", help="The name of the author", default="A.J. Hasenack")
parser.add_argument(
    "--theory", help="The theory used, formatted as 'theory_'+int", default="theory_800"
)
parser.add_argument("--output", help="The name of the output folder", default="results")
args = parser.parse_args()

# Extract command line arguments
pdf_name = args.pdf
folder_name = args.folder
data = args.data
author = args.author
theory = args.theory
output = args.output

# Get PineAPPL grid paths and PDF name
file = get_file(folder_name, data, theory)

# Get predictions and save data
df = get_prediction(file, pdf_name)
save_data(df, data, pdf_name, author, theory, output)
