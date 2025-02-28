"""CLI entry point to check compatibility."""

from dataclasses import dataclass
from enum import Enum

import eko
import pineappl
import rich
import rich_click as click

from .. import check
from ._base import command


@command.group("check")
def subcommand():
    """Check grid and operator properties."""


@subcommand.command("compatibility")
@click.argument("grid_path", metavar="PINEAPPL", type=click.Path(exists=True))
@click.argument("operator_path", metavar="EKO", type=click.Path(exists=True))
@click.option("--xif", default=1.0, help="factorization scale variation")
@click.option("--max_as", type=int, default=5, help="Maximum order of alpha_s to check")
@click.option("--max_al", type=int, default=5, help="Maximum order of alpha to check")
def sub_compatibility(grid_path, operator_path, xif, max_as, max_al):
    """Check PineAPPL grid and EKO compatibility.

    In order to be compatible, the grid provided in PINEAPPL and the operator
    provided in EKO, have to expose the same x grid and Q2 grid.

    XIF is the factorization scale variation.

    max_as and max_al default to a very high value so the comparison is done at the level
    of the entire grid, with no orders masked.
    If only some orders are required the user must use the MAX_AS and MAX_AL flags.

    """
    pineappl_grid = pineappl.grid.Grid.read(grid_path)
    with eko.EKO.read(operator_path) as operators:
        try:
            check.check_grid_and_eko_compatible(
                pineappl_grid, operators, xif, max_as, max_al
            )
            rich.print("[green]Success:[/] grids are compatible")
        except ValueError as e:
            rich.print("[red]Error:[/]", e)


@dataclass
class CouplingInfo:
    """Coupling known attributes, used to describe it."""

    descr: str
    theory: str


class Coupling(Enum):
    """Auxiliary class to list the possible couplings."""

    AS = CouplingInfo("strong", "QCD")
    AL = CouplingInfo("electromagnetic", "QED")


SCVAR_ERROR = "[red]Error:[/] grids do not contain"

SCVAR_MESSAGES = {
    check.AvailableAtMax.BOTH: "[green]Success:[/] grids contain",
    check.AvailableAtMax.CENTRAL: "[orange]Warning:[/] grids do not contain central order for requested",
    check.AvailableAtMax.SCVAR: SCVAR_ERROR,
}


@subcommand.command("scvar")
@click.argument("grid_path", metavar="PINEAPPL", type=click.Path(exists=True))
@click.argument(
    "scale",
    metavar="SCALE",
    type=click.Choice(list(el.name for el in check.Scale), case_sensitive=False),
)
@click.argument("max_as_order", metavar="AS_ORDER", type=int)
@click.argument("max_al_order", metavar="AL_ORDER", type=int)
def sub_scvar(grid_path, scale, max_as_order, max_al_order):
    """Check if PineAPPL grid contains requested scale variations for the requested order."""
    grid = pineappl.grid.Grid.read(grid_path)
    grid.optimize()

    # Call the function
    scaleobj = check.Scale[scale]
    checkres, max_as_effective = check.contains_sv(
        grid, max_as_order, max_al_order, scaleobj
    )

    # Communicate result
    message = SCVAR_MESSAGES[checkres]
    if not max_as_effective == max_as_order:
        message = SCVAR_ERROR
    descr = check.Scale[scale].value.description
    cname = Coupling.AS.name.lower()

    rich.print(f"{message} {descr} for {cname}")
