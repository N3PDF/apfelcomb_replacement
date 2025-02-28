"""Tools related to evolution/eko."""

import copy
import json
import logging
import os
import pathlib
from importlib import metadata
from typing import Optional, Union

import eko
import eko.basis_rotation as br
import numpy as np
import pineappl
import rich
import rich.box
import rich.panel
import yaml
from eko import basis_rotation
from eko.io.types import ScaleVariationsMethod
from eko.matchings import Atlas, nf_default
from eko.quantities import heavy_quarks

from . import check, comparator, version

logger = logging.getLogger(__name__)


def sv_scheme(tcard):
    """Infere the factorization scale_variation scheme to be used from the theory card.

    Parameters
    ----------
    tcard : dict
        theory card

    """
    modsv_list = {a.value for a in ScaleVariationsMethod}
    xif = tcard["XIF"]
    modsv = tcard["ModSV"]
    if np.isclose(xif, 1.0):
        if modsv in modsv_list:
            raise ValueError("ModSv is not None but xif is 1.0")
        return None
    # scheme C case
    if modsv not in modsv_list:
        return None
    return modsv


def get_convolution_suffix(convolution: pineappl.convolutions.Conv) -> str:
    """Get the correct suffix for a given convolution.

    Accounts for all possible combinations of (PDF, FF)⊗(Unpolarized, Polarized).
    If the convolution is PDF⊗Unpolarized (which is the default in NNPDF), then
    the suffix is an empty string (for legacy purposes).

    Parameters
    ----------
    convolution: pineappl.convolutions.Conv
        a PineAPPL object containing the information on the convolution

    Returns
    -------
    suffix: str
        string containing the correct suffix
    """
    suffix = ""
    if convolution.convolution_types.polarized:
        suffix += "_polarized"
    if convolution.convolution_types.time_like:
        suffix += "_time_like"
    return suffix


def write_operator_card_from_file(
    pineappl_path: os.PathLike,
    default_card_path: os.PathLike,
    card_path: os.PathLike,
    tcard: dict,
):
    """Generate operator card for a grid.

    Parameters
    ----------
    pineappl_path : str or os.PathLike
        path to grid to evolve
    default_card : str or os.PathLike
        base operator card
    card_path : str or os.PathLike
        target path
    tcard: dict
        theory card for the run

    Returns
    -------
    x_grid : np.ndarray
        written x grid
    q2_grid : np.ndarray
        written Q2 grid

    """
    # raise in python rather then rust
    if not pathlib.Path(pineappl_path).exists():
        raise FileNotFoundError(pineappl_path)
    pineappl_grid = pineappl.grid.Grid.read(pineappl_path)
    default_card = yaml.safe_load(
        pathlib.Path(default_card_path).read_text(encoding="utf-8")
    )
    return write_operator_card(pineappl_grid, default_card, card_path, tcard)


def dump_card(
    card_path: Union[str, os.PathLike],
    operators_card: dict,
    convolution: pineappl.convolutions.Conv,
) -> None:
    """Set polarization and dump operator cards.

    Parameters
    ----------
    card_path : str or os.PathLike
        target path
    operators_card : dict
        operators card to dump
    convolution : pineappl.convolutions.Conv
        the type of convolution
    """
    op_to_dump = copy.deepcopy(operators_card)
    op_to_dump["configs"]["polarized"] = convolution.convolution_types.polarized
    op_to_dump["configs"]["time_like"] = convolution.convolution_types.time_like

    suffix = get_convolution_suffix(convolution)
    card_path = card_path.parent / f"{card_path.stem}{suffix}.yaml"
    with open(card_path, "w", encoding="UTF-8") as f:
        yaml.safe_dump(op_to_dump, f)
        pineko_version = metadata.version("pineko")
        f.write(f"# {pineko_version=}")

    if card_path.exists():
        rich.print(
            f"[green]Success:[/] Wrote card with {len(operators_card['mugrid'])} Q2 points to {card_path}"
        )


def write_operator_card(
    pineappl_grid: pineappl.grid.Grid,
    default_card: dict,
    card_path: Union[str, os.PathLike],
    tcard: dict,
):
    """Generate operator card for this grid.

    Parameters
    ----------
    pineappl_grid : pineappl.grid.Grid
        grid to evolve
    default_card : dict
        base operator card
    card_path : str or os.PathLike
        target path
    tcard: dict
        theory card for the run, since some information in EKO is now required
        in operator card, but before was in the theory card

    Returns
    -------
    x_grid : np.ndarray
        written x grid
    q2_grid : np.ndarray
        written Q2 grid

    """
    # Add a +1 to the orders for the difference in convention between nnpdf and pineappl
    # NB: This would not happen for nFONLL
    is_fns = int(check.is_fonll_mixed(tcard["FNS"], pineappl_grid.channels()))
    max_as = 1 + tcard["PTO"] + is_fns
    max_al = 1 + tcard["QED"]
    # ... in order to create a mask ...
    order_mask = pineappl.boc.Order.create_mask(
        pineappl_grid.orders(), max_as, max_al, True
    )
    # ... to get the x and muF grids for the eko
    evol_info = pineappl_grid.evolve_info(order_mask)
    muf2_grid = evol_info.fac1
    operators_card = copy.deepcopy(default_card)
    sv_method = sv_scheme(tcard)
    xif = 1.0 if sv_method is not None else tcard["XIF"]
    # update scale variation method
    operators_card["configs"]["scvar_method"] = sv_method

    # Make sure that we are using the theory Q0 and fail if the template has a different one
    operators_card["mu0"] = tcard["Q0"]
    if default_card.get("mu0") is not None and default_card["mu0"] != tcard["Q0"]:
        raise ValueError("Template declares a value of Q0 different from theory")

    q2_grid = (xif * xif * muf2_grid).tolist()
    masses = np.array([tcard["mc"], tcard["mb"], tcard["mt"]]) ** 2
    thresholds_ratios = np.array([tcard["kcThr"], tcard["kbThr"], tcard["ktThr"]]) ** 2
    for q in range(tcard["MaxNfPdf"] + 1, 6 + 1):
        thresholds_ratios[q - 4] = np.inf
    atlas = Atlas(
        matching_scales=heavy_quarks.MatchingScales(masses * thresholds_ratios),
        origin=(tcard["Q0"] ** 2, tcard["nf0"]),
    )
    # If we are producing nFONLL FKs we need to look to NfFF...
    if check.is_num_fonll(tcard["FNS"]):
        nf = tcard["NfFF"]
        operators_card["mugrid"] = [(float(np.sqrt(q2)), int(nf)) for q2 in q2_grid]
    else:
        operators_card["mugrid"] = [
            (float(np.sqrt(q2)), nf_default(q2, atlas)) for q2 in q2_grid
        ]
    if "integrability_version" in pineappl_grid.metadata:
        x_grid = evol_info.x1
        x_grid = np.append(x_grid, 1.0)
        operators_card["configs"]["interpolation_polynomial_degree"] = 1
        operators_card["xgrid"] = x_grid.tolist()

    # Add the version of eko and pineko to the operator card
    # using importlib.metadata.version to get the correct tag in editable mode
    operators_card["eko_version"] = metadata.version("eko")

    # Choose the evolution method according to the theory if the key is included
    if "ModEv" in tcard:
        opconf = operators_card["configs"]
        if tcard["ModEv"] == "TRN":
            opconf["evolution_method"] = "truncated"
            opconf["ev_op_iterations"] = 1
        elif tcard["ModEv"] == "EXA":
            opconf["evolution_method"] = "iterate-exact"
            if "IterEv" in tcard:
                opconf["ev_op_iterations"] = tcard["IterEv"]
            elif "ev_op_iterations" not in default_card["configs"]:
                raise ValueError(
                    "EXA used but IterEv not found in the theory card and not ev_op_iterations set in the template"
                )

        # If the evolution method is defined in the template and it is different, fail
        template_method = default_card["configs"].get("evolution_method")
        if (
            template_method is not None
            and template_method != opconf["evolution_method"]
        ):
            raise ValueError(
                f"The template and the theory have different evolution method ({template_method} vs {opconf['key']})"
            )

        # If the change is on the number of iterations, take the template value but warn the user
        template_iter = default_card["configs"].get("ev_op_iterations")
        if template_iter is not None and template_method != opconf["ev_op_iterations"]:
            opconf["ev_op_iterations"] = template_iter
            logger.warning(
                f"The number of iteration in the theory and template is different, using template value ({template_iter})"
            )

    # Some safety checks
    if (
        operators_card["configs"]["evolution_method"] == "truncated"
        and operators_card["configs"]["ev_op_iterations"] > 1
    ):
        logger.warning(
            "Warning! You are setting evolution_method=truncated with ev_op_iterations>1,"
            "are you sure that's what you want?"
        )

    # Get the types of convolutions required for this Grid
    convolutions = pineappl_grid.convolutions

    for conv in convolutions:
        dump_card(card_path, operators_card, conv)

    return operators_card["xgrid"], q2_grid


def evolve_grid(
    grid: pineappl.grid.Grid,
    operators: list,
    fktable_path: str,
    max_as: int,
    max_al: int,
    xir: float,
    xif: float,
    xia: float,
    assumptions="Nf6Ind",
    comparison_pdfs: Optional[list[str]] = None,
    meta_data=None,
    min_as=None,
):
    """Convolute grid with EKO from file paths.

    Parameters
    ----------
    grid : pineappl.grid.Grid
        unconvolved grid
    operators : list(eko.EKO)
        list of evolution operators
    fktable_path : str
        target path for convolved grid
    max_as : int
        maximum power of strong coupling
    max_al : int
        maximum power of electro-weak coupling
    xir : float
        renormalization scale variation
    xif : float
        factorization scale variation
    xia : float
        fragmentation scale variation
    assumptions : str
        assumptions on the flavor dimension
    comparison_pdfs : list(str) or None
        if given, a comparison table (with / without evolution) will be printed
    meta_data : None or dict
        if given, additional meta data written to the FK table
    min_as: None or int
        minimum power of strong coupling
    """
    order_mask = pineappl.boc.Order.create_mask(grid.orders(), max_as, max_al, True)
    if min_as is not None and min_as > 1:
        # If using min_as, we want to ignore only orders below that (e.g., if min_as=2
        # and max_as=3, we want NNLO and NLO)
        ignore_orders = pineappl.boc.Order.create_mask(
            grid.orders(), min_as - 1, max_al, True
        )
        order_mask ^= ignore_orders

    evol_info = grid.evolve_info(order_mask)
    x_grid = evol_info.x1
    mur2_grid = evol_info.ren1
    # TODO: Find a better way to do this
    xif = 1.0 if operators[0].operator_card.configs.scvar_method is not None else xif
    tcard = operators[0].theory_card
    opcard = operators[0].operator_card
    # rotate the targetgrid
    if "integrability_version" in grid.metadata:
        x_grid = np.append(x_grid, 1.0)

    def xgrid_reshape(full_operator):
        """Reinterpolate operators on output and/or input grids."""
        eko.io.manipulate.xgrid_reshape(
            full_operator, targetgrid=eko.interpolation.XGrid(x_grid)
        )
        check.check_grid_and_eko_compatible(grid, full_operator, xif, max_as, max_al)
        # rotate to evolution (if doable and necessary)
        if np.allclose(full_operator.bases.inputpids, br.flavor_basis_pids):
            eko.io.manipulate.to_evol(full_operator)
        # Here we are checking if the EKO contains the rotation matrix (flavor to evol)
        elif not np.allclose(
            full_operator.bases.inputpids, br.rotate_flavor_to_evolution
        ):
            raise ValueError("The EKO is neither in flavor nor in evolution basis.")

    for operator in operators:
        xgrid_reshape(operator)

    # PineAPPL wants alpha_s = 4*pi*a_s
    # remember that we already accounted for xif in the opcard generation
    evmod = eko.couplings.couplings_mod_ev(opcard.configs.evolution_method)
    # Couplings ask for the square of the masses
    thresholds_ratios = np.power(tcard.heavy.matching_ratios, 2.0)
    for q in range(tcard.couplings.max_num_flavs + 1, 6 + 1):
        thresholds_ratios[q - 4] = np.inf
    sc = eko.couplings.Couplings(
        tcard.couplings,
        tcard.order,
        evmod,
        masses=[(x.value) ** 2 for x in tcard.heavy.masses],
        hqm_scheme=tcard.heavy.masses_scheme,
        thresholds_ratios=thresholds_ratios.tolist(),
    )
    # To compute the alphas values we are first reverting the factorization scale shift
    # and then obtaining the renormalization scale using xir.
    ren_grid2 = xir * xir * mur2_grid
    alphas_values = [
        4.0
        * np.pi
        * sc.a_s(
            mur2,
        )
        for mur2 in ren_grid2
    ]

    def prepare(operator, convolution_types):
        """Match the raw operator with its relevant metadata."""
        # TODO: This is the most important part to check. In any case, this would better
        # bet an iterator even if it doesn't provide improvements
        sub_slices = []
        for (q2, _), op in operator.items():
            info = pineappl.evolution.OperatorSliceInfo(
                fac0=operator.mu20,
                fac1=q2,
                x0=operator.bases.inputgrid.raw,
                x1=operator.bases.targetgrid.raw,
                pids0=basis_rotation.evol_basis_pids,
                pids1=operator.bases.targetpids,
                pid_basis=pineappl.pids.PidBasis.Evol,
                convolution_types=convolution_types,
            )
            sub_slices.append((info, op.operator))
        return sub_slices

    # TODO: Check compatibility between EKOs and CONVs and recombine similar convolutions
    slices = [
        prepare(o, c.convolution_types) for o, c in zip(operators, grid.convolutions)
    ]

    # Perform the arbitrary many evolutions
    fktable = grid.evolve(
        slices=slices,
        order_mask=order_mask,
        xi=(xir, xif, xia),
        ren1=ren_grid2,
        alphas=alphas_values,
    )

    rich.print(f"Optimizing for {assumptions}")
    fktable.optimize(pineappl.fk_table.FkAssumptions(assumptions))
    fktable.set_metadata("eko_version", operators[0].metadata.version)
    fktable.set_metadata("eko_theory_card", json.dumps(operators[0].theory_card.raw))

    for idx, operator in enumerate(operators):
        suffix = "" if idx == 0 else f"_{idx}"
        fktable.set_metadata(
            f"eko_operator_card{suffix}", json.dumps(operator.operator_card.raw)
        )

    fktable.set_metadata("pineko_version", version.__version__)
    if meta_data is not None:
        for k, v in meta_data.items():
            fktable.set_metadata(k, v)

    # compare before/after
    comparison = None
    if comparison_pdfs is not None:
        scales = (xir, xif, xia)
        comparison = comparator.compare(
            grid, fktable, max_as, max_al, comparison_pdfs, scales
        )
        fktable.set_metadata("results_fk", comparison.to_string())
        for idx, pdf in enumerate(comparison_pdfs):
            fktable.set_metadata(f"results_fk_pdfset{idx}", str(pdf))
    # write
    fktable.write_lz4(str(fktable_path))
    return grid, fktable, comparison
