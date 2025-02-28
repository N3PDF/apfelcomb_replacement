"""Module to generate scale variations."""

import pathlib
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pineappl
from eko import beta

from . import check

AS_NORM = 1.0 / (4.0 * np.pi)
OrderTuple = Tuple[int, int, int, int]
"""Tuple representing a PineAPPL order."""


class ReturnState(Enum):
    """Auxiliary class to list the possible return states."""

    ALREADY_THERE = "[green]Renormalization scale variations are already in the grid"
    ORDER_EXISTS_FAILURE = (
        "Order_exists is True but the order does not appear to be in the grid"
    )
    MISSING_CENTRAL = "Central order is not high enough to compute requested sv orders"
    SUCCESS = "[green]Success: scale variation orders included!"


def qcd(order: OrderTuple) -> int:
    """Extract the QCD order from an OrderTuple."""
    return order[0]


def orders_as_tuple(grid: pineappl.grid.Grid) -> List[OrderTuple]:
    """Return grid orders as a tuple."""
    return [order.as_tuple() for order in grid.orders()]


def ren_sv_coeffs(m, max_as, logpart, which_part, nf):
    """Ren_sv coefficient for the requested part.

    Parameters
    ----------
    m : int
        first non zero perturbative order
    max_as : int
        max order of alpha_s
    logpart : int
        power of the renormalization scale log asked
    which_part : int
        asked perturbative order contribution to be rescaled
    nf : int
        number of active flavors

    Returns
    -------
    float
        renormalization scale variation contribution
    """
    # nothing to do?
    if max_as == 0:
        return 0.0
    # eko uses as = alpha_s/(4pi), but pineappl just alpha_s
    as_normalization = AS_NORM ** (max_as - which_part)
    # the coefficients can be found in the NNLO MHOU paper
    # (which also contains a generating MMa script)
    beta0 = beta.beta_qcd_as2(nf)
    beta1 = beta.beta_qcd_as3(nf)
    beta2 = beta.beta_qcd_as4(nf)
    ren_coeffs = {
        # NLO
        (1, 1, 0): m * beta0,
        # NNLO
        (2, 1, 1): (m + 1) * beta0,
        (2, 1, 0): (m + 0) * beta1,
        (2, 2, 0): m * (m + 1) / 2.0 * beta0**2,
        # N3LO
        (3, 1, 2): (m + 2) * beta0,
        (3, 1, 1): (m + 1) * beta1,
        (3, 1, 0): (m + 0) * beta2,
        (3, 2, 1): (m + 1) * (m + 2) / 2.0 * beta0**2,
        (3, 2, 0): m * (2 * m + 3) / 2.0 * beta0 * beta1,
        (3, 3, 0): m * (m + 1) * (m + 2) / 6 * beta0**3,
    }
    return as_normalization * ren_coeffs[(max_as, logpart, which_part)]


def requirements(m: int, max_as: int, al: int) -> Dict[OrderTuple, List[OrderTuple]]:
    """Return a dictionary with the orders required to have the muR scale variation.

    `m` is the first non-zero perturbative order of the grid, and `al` is the QED order
    of the "QCD" leading order.
    """
    return {
        (m + max_as, al, delt + 1, 0, 0): [
            (m + de, al, 0, 0, 0) for de in range(max_as - delt)
        ]
        for delt in range(max_as)
    }


def initialize_new_grid(grid, new_order):
    """Initialize a new grid similar to the original with the `oder` modified."""
    # Retrieve parameters to create new grid. The +1 explanation is that n bins
    # have n+1 bin limits, and range generates numbers from a half-open interval
    # (range(n) generates n numbers).
    bin_limits = [float(bin) for bin in range(grid.bins() + 1)]
    channels = [pineappl.boc.Channel(mychannel) for mychannel in grid.channels()]
    new_order = [pineappl.boc.Order(*new_order)]

    # Construct the bin object
    bin_limits = pineappl.boc.BinsWithFillLimits.from_fill_limits(
        fill_limits=bin_limits
    )

    # create a new grid that is similar to `grid` but with `new_order`
    return pineappl.grid.Grid(
        pid_basis=grid.pid_basis,
        channels=channels,
        orders=new_order,
        bins=bin_limits,
        convolutions=grid.convolutions,
        interpolations=grid.interpolations,
        kinematics=grid.kinematics,
        scale_funcs=grid.scales,
    )


def create_svonly(grid, order, new_order, scalefactor):
    """Create a grid containing only the renormalization scale variations at a given order for a grid."""
    new_grid = initialize_new_grid(grid, new_order)
    # extract the relevant order to rescale from the grid for each lumi and bin
    grid_orders = orders_as_tuple(grid)
    order_index = grid_orders.index(order)
    for lumi_index in range(len(new_grid.channels())):
        for bin_index in range(grid.bins()):
            extracted_subgrid = grid.subgrid(order_index, bin_index, lumi_index)
            extracted_subgrid.scale(scalefactor)
            # Set this subgrid inside the new grid
            new_grid.set_subgrid(0, bin_index, lumi_index, extracted_subgrid)
    # Fixing bin_limits and normalizations
    bin_dimension = grid.bin_dimensions()
    limits = []
    bin_specs = np.array(grid.bin_limits())
    for num_bin in range(grid.bins()):
        for dim in range(bin_dimension):
            bin_left = bin_specs[:, dim, 0][num_bin]
            bin_right = bin_specs[:, dim, 1][num_bin]
            limits.append([(bin_left, bin_right)])
    norma = grid.bin_normalizations()
    bin_configs = pineappl.boc.BinsWithFillLimits.from_limits_and_normalizations(
        limits=limits,
        normalizations=norma,
    )
    new_grid.set_bwfl(bin_configs)
    return new_grid


def create_grids(gridpath, max_as, nf):
    """Create all the necessary scale variations grids for a certain starting grid."""
    grid = pineappl.grid.Grid.read(gridpath)
    grid_orders = orders_as_tuple(grid)
    order_mask = pineappl.boc.Order.create_mask(grid.orders(), max_as, 0, True)
    grid_orders_filtered = list(np.array(grid_orders)[order_mask])
    grid_orders_filtered.sort(key=qcd)
    first_nonzero_order = grid_orders_filtered[0]
    min_al = first_nonzero_order[1]
    m_value = first_nonzero_order[0]
    nec_orders = requirements(m_value, max_as, min_al)
    grid_list = {}
    for to_construct_order in nec_orders:
        list_grid_order = []
        for nec_order in nec_orders[to_construct_order]:
            # The logpart of the coefficient I am asking is just the [2] entry of to_construct_order
            # The QCD order of the part I am rescaling is just nec_order[0] but I need to rescale it with respect to the first non-zero order
            scalefactor = ren_sv_coeffs(
                m_value, max_as, to_construct_order[2], nec_order[0] - m_value, nf
            )
            list_grid_order.append(
                create_svonly(grid, nec_order, to_construct_order, scalefactor)
            )
        grid_list[to_construct_order] = list_grid_order

    return grid_list


def construct_and_dump_order_exists_grid(ori_grid, to_construct_order):
    """Remove the order that has to be substituted from the grid.

    Parameters
    ----------
    ori_grid:
        original grid
    to_construct_order:
        order to delete
    """
    # TODO: can we make this function simpler ??
    bin_limits = [float(bin) for bin in range(ori_grid.bins() + 1)]
    lumi_grid = [pineappl.boc.Channel(mylum) for mylum in ori_grid.channels()]
    subgrid_params = pineappl.subgrid.SubgridParams()
    ori_grid_orders = orders_as_tuple(ori_grid)
    new_orders = [
        pineappl.grid.Order(*ord)
        for ord in ori_grid_orders
        if ord != to_construct_order
    ]
    new_grid = pineappl.grid.Grid.create(
        lumi_grid, new_orders, bin_limits, subgrid_params
    )
    orders_indices = [ori_grid_orders.index(order.as_tuple()) for order in new_orders]
    for order_index in orders_indices:
        for lumi_index in range(len(lumi_grid)):
            for bin_index in range(ori_grid.bins()):
                extr_subgrid = ori_grid.subgrid(order_index, bin_index, lumi_index)
                new_grid.set_subgrid(
                    orders_indices.index(order_index),
                    bin_index,
                    lumi_index,
                    extr_subgrid,
                )
    bin_dimension = ori_grid.bin_dimensions()
    limits = []
    for num_bin in range(ori_grid.bins()):
        for dim in range(bin_dimension):
            limits.append(
                (
                    ori_grid.bin_left(dim)[num_bin],
                    ori_grid.bin_right(dim)[num_bin],
                )
            )
    norma = ori_grid.bin_normalizations()
    bin_configs = pineappl.boc.BinsWithFillLimits.from_limits_and_normalizations(
        limits=limits,
        normalizations=norma,
    )
    new_grid.set_bwfl(bin_configs)

    # propagate metadata
    for k, v in ori_grid.metadata.items():
        new_grid.set_metadata(k, v)

    return new_grid


def compute_ren_sv_grid(
    grid_path: pathlib.Path,
    max_as: int,
    nf: int,
    target_path: Optional[pathlib.Path] = None,
    order_exists: bool = False,
):
    """Generate renormalization scale variation terms for the given grid, according to the max_as."""
    # First let's check if the ren_sv are already there
    grid = pineappl.grid.Grid.read(grid_path)
    checkres, max_as_effective = check.contains_sv(grid, max_as, 0, check.Scale.REN)
    # Usual different convention with max_as
    if max_as_effective == max_as and (checkres is not check.AvailableAtMax.CENTRAL):
        if not order_exists:
            return ReturnState.ALREADY_THERE
    elif order_exists:
        return ReturnState.ORDER_EXISTS_FAILURE
    if max_as_effective < max_as and checkres is check.AvailableAtMax.SCVAR:
        return ReturnState.MISSING_CENTRAL
    # Create all the necessary grids
    # With respect to the usual convention here max_as is max_as-1
    grid_list = create_grids(grid_path, max_as - 1, nf)

    # merge them
    for sv_order, sv_grids in grid_list.items():
        # if the new order is there, clean the old one.
        if sv_order in orders_as_tuple(grid):
            grid = construct_and_dump_order_exists_grid(grid, list(grid_list)[0])
        for sv_grid in sv_grids:
            grid.merge(sv_grid)
    # save
    if target_path is None:
        target_path = grid_path.parent
    grid.write_lz4(target_path / grid_path.name)
    return ReturnState.SUCCESS
