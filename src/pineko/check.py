"""Tools to check compatibility of EKO and grid."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import numpy as np
import pineappl


@dataclass
class ScaleValue:
    """Contain the information of a kind of scale variations and its index in the orders of a pineappl grid."""

    description: str
    index: int


class Scale(Enum):
    """Auxiliary class to list the possible scale variations."""

    REN = ScaleValue("renormalization scale variations", -3)
    FACT = ScaleValue("factorization scale variations", -2)
    FRAG = ScaleValue("fragmentation scale variation", -1)


class AvailableAtMax(Enum):
    """Hold the information of a scale variation check.

    BOTH means that both the central order and the scale variation order are contained in the grid.
    CENTRAL means that only the central order is present.
    SCVAR means that only the scale variation order is present.

    """

    BOTH = auto()
    CENTRAL = auto()
    SCVAR = auto()


def in1d(a, b, rtol=1e-05, atol=1e-08):
    """Improved version of np.in1d.

    Thanks: https://github.com/numpy/numpy/issues/7784#issuecomment-848036186

    Parameters
    ----------
    a : list
        needle
    b : list
        haystack
    rtol : float
        allowed relative error
    atol : float
        allowed absolute error

    Returns
    -------
    list
        mask of found elements
    """
    if len(a) == 1:
        for be in b:
            if np.isclose(be, a[0], rtol=rtol, atol=atol):
                return [True]
        return [False]
    ss = np.searchsorted(a[1:-1], b, side="left")
    return np.isclose(a[ss], b, rtol=rtol, atol=atol) | np.isclose(
        a[ss + 1], b, rtol=rtol, atol=atol
    )


def check_grid_and_eko_compatible(pineappl_grid, operators, xif, max_as, max_al):
    """Check whether the EKO operators and the PineAPPL grid are compatible.

    Parameters
    ----------
    pineappl_grid : pineappl.grid.Grid
        grid
    operators : eko.EKO
        operators
    xif : float
        factorization scale variation
    max_as: int
        max order of alpha_s
    max_al: int
        max order of alpha

    Raises
    ------
    ValueError
        If the operators and the grid are not compatible.
    """
    order_mask = pineappl.boc.Order.create_mask(
        pineappl_grid.orders(), max_as, max_al, True
    )
    evol_info = pineappl_grid.evolve_info(order_mask)
    x_grid = evol_info.x1
    muf2_grid = evol_info.fac1
    # Q2 grid
    if not np.all(
        in1d(np.unique(list(operators.mu2grid)), xif * xif * np.array(muf2_grid))
    ):
        raise ValueError(
            "Q2 grid in pineappl grid and eko operator are NOT compatible!"
        )
    # x-grid
    if not np.all(
        in1d(np.unique(operators.bases.targetgrid.tolist()), np.array(x_grid))
    ):
        raise ValueError("x grid in pineappl grid and eko operator are NOT compatible!")


def is_dis(channels):
    """Check if the fktable we are computing is a DIS fktable.

    Parameters
    ----------
    channels : list(list(tuple(list, float)))
        object containing the information on the channels. From PineAPPL v1,
        the tuple contains two elements. The first is a list containing the
        PIDs of the partons. The length of the list coincides with the number
        of partons involved, ie for DIS there is only one element, and for a
        hadron production in pp collision there are three elements. The 2nd
        element is the weight/multiplicative factor.

    Returns
    -------
    bool
        true if the fktable is a DIS fktable
    """
    return True if len(channels[0][0][0]) == 1 else False


def is_fonll_mixed(fns, lumi):
    """Check if the fktable we are computing is FONLL-B, FONLL-D or, in general, a mixed FONLL fktable.

    Parameters
    ----------
    fns : str
          flavor number scheme (from the theory card)
    lumi : list(list(tuple))
           luminosity info

    Returns
    -------
    bool
        true if the fktable is a mixed FONLL DIS fktable
    """
    return is_dis(lumi) and fns in ["FONLL-B", "FONLL-D"]


def is_num_fonll(fns):
    """Check if the FNS is a nFONLL FNS."""
    return fns in ["FONLL-FFNS", "FONLL-FFN0"]


def orders(grid, max_as, max_al) -> list:
    """Select the relevant orders.

    The orders in the grid are filtered according to `max_as` and `max_al`.

    """
    order_array = np.array([order.as_tuple() for order in grid.orders()])
    order_mask = pineappl.boc.Order.create_mask(grid.orders(), max_as, max_al, True)
    order_list = order_array[order_mask]
    return order_list


def pure_qcd(orders):
    """Select the QCD LO and pure QCD corrections to it."""
    min_al = min(ord[1] for ord in orders)
    return [ord for ord in orders if ord[1] == min_al]


def contains_sv(
    grid: pineappl.grid.Grid, max_as: int, max_al: int, sv_type: Scale
) -> Tuple[AvailableAtMax, int]:
    """Check whether renormalization scale-variations are available in the pineappl grid."""
    svindex = sv_type.value.index
    ords = pure_qcd(orders(grid, max_as, max_al))
    max_as = max(ord[0] for ord in ords)
    min_as = min(ord[0] for ord in ords)
    max_as_cen = max(ord[0] for ord in ords if ord[svindex] == 0)
    max_as_sv = max((ord[0] for ord in ords if ord[svindex] != 0), default=0)
    if max_as_cen == max_as:
        if max_as_sv == max_as:
            checkres = AvailableAtMax.BOTH
        # This is the LO case so for both FACT and REN we do not expect sv orders at all
        elif max_as == min_as:
            checkres = AvailableAtMax.BOTH
        # For renormalization scale variations, the NLO sv order is not present if the first non zero order is at alpha^0
        elif max_as == 1 and sv_type is Scale.REN and min_as == 0:
            checkres = AvailableAtMax.BOTH
        else:
            checkres = AvailableAtMax.CENTRAL
    else:
        checkres = AvailableAtMax.SCVAR
    # Since max_as_effective will be compared to max_as and we are using different conventions for the two, here we sum 1 to max_as_effective and make it relative to the first non zero order
    return checkres, max_as - min_as + 1
