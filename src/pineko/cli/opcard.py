"""CLI entry point to the operator card generation."""

import pathlib

import rich_click as click
import yaml

from .. import evolve
from ._base import command


@command.command("opcard")
@click.argument("pineappl-path", metavar="PINEAPPL", type=click.Path(exists=True))
@click.argument(
    "default-card-path", metavar="DEFAULT_CARD", type=click.Path(exists=True)
)
@click.argument("thcard-path", metavar="THCARD", type=click.Path())
@click.argument("opcard-path", metavar="OPCARD", type=click.Path())
def subcommand(pineappl_path, default_card_path, thcard_path, opcard_path):
    """Write EKO card for PineAPPL grid.

    Writes a copy of the default card from DEFAULT_CARD to OPCARD
    with the adjusted x grid and Q2 grid read from PINEAPPL.

    A THCARD is required, since some of the EKO's OPCARD information come from
    the NNPDF theory entries (e.g. :math:`Q0`).

    """
    tcard = yaml.safe_load(pathlib.Path(thcard_path).read_text(encoding="utf-8"))
    opcard_path = pathlib.Path(opcard_path)
    _x_grid, q2_grid = evolve.write_operator_card_from_file(
        pineappl_path, default_card_path, opcard_path, tcard
    )
