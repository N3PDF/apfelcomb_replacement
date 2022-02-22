# -*- coding: utf-8 -*-
import eko
import eko.basis_rotation as br
import numpy as np
import pineappl
import rich
import rich.box
import rich.panel

from . import check, comparator


def evolve_grid(
    pineappl_path, eko_path, fktable_path, max_as, max_al, comparison_pdf=None
):
    """
    Invoke steps from file paths.

    Parameters
    ----------
        pineappl_path : str
            unconvoluted grid
        eko_path : str
            evolution operator
        fktable_path : str
            target path for convoluted grid
        max_as : int
            maximum power of strong coupling
        max_al : int
            maximum power of electro-weak coupling
        comparison_pdf : None or str
            if given, a comparison table (with / without evolution) will be printed
    """
    rich.print(
        rich.panel.Panel.fit(f"Computing ...", style="magenta", box=rich.box.SQUARE),
        f"   {pineappl_path}\n",
        f"+ {eko_path}\n",
        f"= {fktable_path}",
    )
    # load
    pineappl_grid = pineappl.grid.Grid.read(str(pineappl_path))
    operators = eko.output.Output.load_tar(eko_path)
    check.check_grid_and_eko_compatible(pineappl_grid, operators)
    # rotate to evolution (if doable and necessary)
    if np.allclose(operators["inputpids"], br.flavor_basis_pids):
        operators.to_evol()
    elif not np.allclose(operators["inputpids"], br.evol_basis_pids):
        raise ValueError("The EKO is neither in flavor nor in evolution basis.")
    # do it
    order_mask = pineappl.grid.Order.create_mask(pineappl_grid.orders(), max_as, max_al)
    fktable = pineappl_grid.convolute_eko(operators, "evol", order_mask=order_mask)
    # write
    fktable.write_lz4(str(fktable_path))
    # compare before after
    if comparison_pdf is not None:
        print(
            comparator.compare(
                pineappl_grid, fktable, max_as, max_al, comparison_pdf
            ).to_string()
        )
