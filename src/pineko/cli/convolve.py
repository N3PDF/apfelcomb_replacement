"""CLI entry point to convolution."""

import eko
import pineappl
import rich
import rich_click as click

from .. import evolve
from ._base import command


@command.command("convolve")
@click.argument("fktable", type=click.Path())
@click.argument("grid_path", type=click.Path(exists=True))
@click.argument("max_as", type=int)
@click.argument("max_al", type=int)
@click.argument("op_paths", type=click.Path(exists=True), nargs=-1)
@click.option("--xir", default=1.0, help="renormalization scale variation")
@click.option("--xif", default=1.0, help="factorization scale variation")
@click.option("--xia", default=1.0, help="fragmentation scale variation")
@click.option("--min_as", type=int, help="Minimum exponent of as")
@click.option(
    "--pdfs",
    default=None,
    type=click.STRING,
    help="List of PDF sets passed as a string, separated by a comma",
    show_default=True,
)
@click.option(
    "--assumptions",
    default="Nf6Ind",
    help="the flavor assumptions to be used",
    show_default=True,
)
def subcommand(
    fktable,
    grid_path,
    max_as,
    max_al,
    op_paths,
    xir,
    xif,
    xia,
    pdfs,
    assumptions,
    min_as,
):
    """Convolute PineAPPL grid and EKO into an FK table.

    GRID_PATH and OP_PATH are the path to the respective elements to convolve.
    Note that multiple operators are allowed.
    FKTABLE is the path where to dump the output.

    MAX_AS and MAX_AL are used to specify the order in QCD and QED
    couplings (i.e. the maximum power allowed for each correction)
    e.g., max_as = 1, max_al = 0 would select LO QCD only
    max_as = 3 instead would select LO, NLO, NNLO QCD.
    While, by default, all orders below MAX_AS will be selected, it is also possible
    to selected only from a certain order. E.g., to get only NNLO & NLO contributions
    it is possible to set max_as = 3, min_as = 2.

    XIR, XIF, and XIA represent the renormalization, factorization, and fragmentation
    scale in the grid, respectively.

    ASSUMPTIONS represent the assumptions on the flavor dimension.

    PDFS is an optional argument. If not None it should be a list containing the names
    of the PDF sets, ordered in the same as the convolution types in the GRID/FK table.
    The PDFs are passed as strings with the names separated by commas.
    """
    grid = pineappl.grid.Grid.read(grid_path)
    n_ekos = len(op_paths)
    with eko.EKO.edit(op_paths[0]) as first_operator:
        operators = [first_operator]
        path_operators = f"[+] {op_paths[0]}\n"
        # If there are more than ONE operator, then account for all of them.
        if len(operators) > 1:
            for op_idx in range(1, n_ekos):
                operators.append(eko.EKO.edit(op_paths[op_idx]))
                path_operators += f"[+] {op_paths[1]}\n"

        rich.print(
            rich.panel.Panel.fit("Computing ...", style="magenta", box=rich.box.SQUARE),
            f"   {grid_path}\n",
            f"{path_operators}",
            f"= {fktable}\n",
            f"with max_as={max_as}, max_al={max_al}, xir={xir}, xif={xif}, xia={xia}",
            f"min_as: {min_as}" if min_as is not None else "",
        )

        pdfs = pdfs.split(",") if pdfs is not None else pdfs
        _grid, _fk, comp = evolve.evolve_grid(
            grid,
            operators,
            fktable,
            max_as,
            max_al,
            xir,
            xif,
            xia,
            assumptions=assumptions,
            comparison_pdfs=pdfs,
            min_as=min_as,
        )

        if len(operators) > 1:
            for op in operators[1:]:
                op.close()

        if comp is not None:
            print(comp.to_string())
