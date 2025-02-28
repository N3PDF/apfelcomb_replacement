"""CLI entry point to generation of scale variations from central grid."""

import pathlib

import rich
import rich_click as click

from .. import scale_variations
from ._base import command


@command.command("ren_sv_grid")
@click.argument("pineappl_path", type=click.Path(exists=True))
@click.argument("target_path", type=click.Path(exists=False))
@click.argument("max_as", type=int)
@click.argument("nf", type=int)
@click.argument("order_exists", type=bool)
def ren_sv_grid(pineappl_path, target_path, max_as, nf, order_exists):
    """Construct new grid with renormalization scale variations included."""
    return_state = scale_variations.compute_ren_sv_grid(
        pathlib.Path(pineappl_path),
        max_as,
        nf,
        target_path=pathlib.Path(target_path),
        order_exists=order_exists,
    )
    rich.print(return_state)
