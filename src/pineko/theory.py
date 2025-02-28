"""Tools related to generation of a list of FK tables.

The typical use case of pineko is the generation of a list of FK tables,
all with common theory parameters. The collective list of this FK tables
together with other theory ingredients (such as C-factors) are often
commonly referred to as 'theory'.
"""

import json
import logging
import time

import eko
import eko.io.legacy
import numpy as np
import pineappl
import rich
import yaml
from eko.runner.managed import solve

from . import check, configs, evolve, parser, scale_variations, theory_card
from .utils import read_grids_from_nnpdf

logger = logging.getLogger(__name__)


def get_eko_names(grid_path, name, filter=True):
    """Get the names of the ekos depending on the types of convolutions.

    Parameters
    ----------
    grid_path : pathlib.Path
        path to grid
    name : str
        grid name, i.e. it's true stem
    filter: bool
        if True removes the duplicates in the list

    Returns
    -------
    list[str] :
         list containing the names of the ekos
    """
    grid_kv = pineappl.grid.Grid.read(grid_path)
    convolutions = grid_kv.convolutions
    names = []
    for convolution in convolutions:
        suffix = evolve.get_convolution_suffix(convolution)
        names.append(f"{name}{suffix}")
    if filter:
        eko_names = list(set(names))
    else:
        # If the types of convolutions are the same, then always return one element
        eko_names = [names[0]] if len(set(names)) == 1 else names
    return eko_names


def check_scvar_evolve(grid, max_as, max_al, kind: check.Scale):
    """Check scale variations and central orders consistency."""
    available, max_as_effective = check.contains_sv(grid, max_as, max_al, kind)
    if max_as == max_as_effective:
        if available is check.AvailableAtMax.SCVAR:
            raise ValueError("Central order is not available but sv order is.")
    if max_as < max_as_effective and available is not check.AvailableAtMax.BOTH:
        raise ValueError("No available central order or sv order.")


class TheoryBuilder:
    """Common builder application to create the ingredients for a theory.

    Parameters
    ----------
    theory_id : int
        theory identifier
    datsets : list(str)
        list of datasets
    silent : bool
        suppress logs
    clear_logs : bool
        erease previos logs (instead of appending)
    overwrite : bool
        allow files to be overwritten instead of skipping
    """

    def __init__(
        self, theory_id, datasets, silent=False, clear_logs=False, overwrite=False
    ):
        """Initialize theory object."""
        self.theory_id = theory_id
        self.datasets = datasets
        self.silent = silent
        self.clear_logs = clear_logs
        self.overwrite = overwrite

    @property
    def operator_cards_path(self):
        """Suffix paths.operator_cards with theory id."""
        return configs.configs["paths"]["operator_cards"] / str(self.theory_id)

    def ekos_path(self, tid=None):
        """Suffix paths.ekos with theory id.

        Parameters
        ----------
        tid : int
            theory id, defaults to my theory id

        Returns
        -------
        pathlib.Path :
            true path
        """
        if tid is None:
            tid = self.theory_id
        return configs.configs["paths"]["ekos"] / str(tid)

    @property
    def fks_path(self):
        """Suffix paths.fktables with theory id."""
        return configs.configs["paths"]["fktables"] / str(self.theory_id)

    def grids_path(self, tid=None):
        """Suffix paths.grids with theory id.

        Parameters
        ----------
        tid : int
            theory id, defaults to my theory id

        Returns
        -------
        pathlib.Path :
            true path
        """
        if tid is None:
            tid = self.theory_id
        return configs.configs["paths"]["grids"] / str(tid)

    def load_grids(self, ds):
        """Load all grids (i.e. process scale) of a dataset.

        Parameters
        ----------
        ds : str
            dataset name

        Returns
        -------
        grids : dict
            mapping basename to path
        """
        # Take fktable information from NNPDF
        raw_grids = read_grids_from_nnpdf(ds, configs.configs)
        if raw_grids is not None:
            grids = [self.grids_path() / i for i in raw_grids]
        else:
            paths = configs.configs["paths"]
            _info, raw_grids = parser.get_yaml_information(
                paths["ymldb"] / f"{ds}.yaml", self.grids_path()
            )
            # the list is still nested, so flatten
            grids = [grid for opgrids in raw_grids for grid in opgrids]

        # then turn into a map name -> path
        grids = {grid.stem.rsplit(".", 1)[0]: grid for grid in grids}
        return grids

    def inherit_grid(self, name, grid, other):
        """Inherit a grid to a new theory.

        Parameters
        ----------
        name : str
            grid name, i.e. it's true stem
        grid : pathlib.Path
            path to grid
        other : pathlib.Path
            new folder
        """
        new = other / f"{name}.{parser.EXT}"
        if new.exists():
            if not self.overwrite:
                rich.print(f"Skipping existing grid {new}")
                return
            new.unlink()
        # link
        new.symlink_to(grid)
        if new.exists():
            rich.print(f"[green]Success:[/] Created link at {new}")

    def inherit_grids(self, target_theory_id):
        """Inherit grids to a new theory.

        Parameters
        ----------
        target_theory_id : int
            target theory id
        """
        other = self.grids_path(target_theory_id)
        other.mkdir(exist_ok=True)
        self.iterate(self.inherit_grid, other=other)

    def inherit_eko(self, name, grid, other):
        """Inherit a EKO to a new theory.

        Parameters
        ----------
        name : str
            grid name, i.e. it's true stem
        grid : pathlib.Path
            path to grid
        other : pathlib.Path
            new folder
        """
        names = get_eko_names(grid, name)
        for name in names:
            eko_path = self.ekos_path() / f"{name}.tar"
            new = other / f"{name}.tar"
            if new.exists():
                if not self.overwrite:
                    rich.print(f"Skipping existing eko {new}")
                    return
                new.unlink()
            # link
            new.symlink_to(eko_path)
            if new.exists():
                rich.print(f"[green]Success:[/] Created link at {new}")

    def inherit_ekos(self, target_theory_id):
        """Inherit ekos to a new theory.

        Parameters
        ----------
        target_theory_id : int
            target theory id
        """
        other = self.ekos_path(target_theory_id)
        other.mkdir(exist_ok=True)
        self.iterate(self.inherit_eko, other=other)

    def iterate(self, f, **kwargs):
        """Iterate grids in datasets.

        Additional keyword arguments are simply passed down.

        Parameters
        ----------
        f : callable
            iterated callable recieving name and grid as argument
        """
        for ds in self.datasets:
            rich.print(f"Analyze {ds}")
            grids = self.load_grids(ds)
            for name, grid in grids.items():
                f(name, grid, **kwargs)
            rich.print()

    def opcard(self, name, grid, tcard):
        """Write a single operator card.

        Parameters
        ----------
        name : str
            grid name, i.e. it's true stem
        grid : pathlib.Path
            path to grid
        tcard : dict
            theory card
        """
        opcard_path = self.operator_cards_path / f"{name}.yaml"
        if opcard_path.exists():
            if not self.overwrite:
                rich.print(f"Skipping existing operator card {opcard_path}")
                return
        _x_grid, q2_grid = evolve.write_operator_card_from_file(
            grid,
            self.operator_cards_path
            / configs.configs["paths"]["operator_card_template_name"],
            opcard_path,
            tcard,
        )

    def opcards(self):
        """Write operator cards."""
        tcard = theory_card.load(self.theory_id)
        self.operator_cards_path.mkdir(exist_ok=True)
        self.iterate(self.opcard, tcard=tcard)

    def load_operator_card(self, name):
        """Read current operator card.

        Parameters
        ----------
        name : str
            grid name, i.e. it's true stem

        Returns
        -------
        ocard : dict
            operator card
        """
        opcard_path = self.operator_cards_path / f"{name}.yaml"
        with open(opcard_path, encoding="utf-8") as f:
            ocard = yaml.safe_load(f)
        return ocard

    def activate_logging(self, path, filename, activated_loggers=()):
        """Activate the logging facilities.

        Parameters
        ----------
        path : pathlib.Path
            source directory
        filename : str
            log file name
        activated_loggers : list(str)
            list of loggers that get registered
        """
        # nothing to do?
        if self.silent or not path:
            return False
        # evtually remove old stuff?
        log_path = path / filename
        if self.clear_logs:
            log_path.write_text("")
        # register everything
        log_file = logging.FileHandler(log_path)
        log_file.setLevel(logging.INFO)
        log_file.setFormatter(
            logging.Formatter("%(asctime)s %(name)s/%(levelname)s: %(message)s")
        )
        for logger_ in (logger, *[logging.getLogger(n) for n in activated_loggers]):
            logger_.handlers = []
            logger_.addHandler(log_file)
            logger_.setLevel(logging.INFO)
        return True

    def eko(self, name, grid, tcard):
        """Compute a single eko.

        Parameters
        ----------
        name : str
            grid name, i.e. it's true stem
        grid : pathlib.Path
            path to grid
        tcard : dict
            theory card
        """
        paths = configs.configs["paths"]
        # activate logging
        self.activate_logging(
            paths["logs"]["eko"], f"{self.theory_id}-{name}.log", ("eko",)
        )
        # setup data
        names = get_eko_names(grid, name)

        for name in names:
            ocard = self.load_operator_card(name)
            # For nFONLL mixed prescriptions (such as FONLL-B) the PTO written on
            # the tcard is used to produce the grid by yadism and it might be different
            # from the PTO needed for the PDF evolution (and so by EKO). Here we
            # ensure that the PTO used in the EKO calculation reflects the real
            # perturbative order of the prescription.
            if tcard.get("PTOEKO") is not None:
                tcard["PTO"] = tcard["PTOEKO"]
            # Deprecated keys still needed by eko below. TODO: remove them asap.
            tcard["Qedref"] = tcard["Qref"]
            tcard["MaxNfAs"] = tcard["MaxNfPdf"]
            # The operator card has been already generated in the correct format
            # The theory card needs to be converted to a format that eko can use
            legacy_class = eko.io.runcards.Legacy(tcard, ocard)
            new_theory = legacy_class.new_theory
            new_op = eko.io.runcards.OperatorCard.from_dict(ocard)
            eko_filename = self.ekos_path() / f"{name}.tar"
            if eko_filename.exists():
                if not self.overwrite:
                    rich.print(f"Skipping existing operator {eko_filename}")
                    return
                eko_filename.unlink()
            # do it!
            logger.info("Start computation of %s", name)
            start_time = time.perf_counter()
            # Actual computation of the EKO
            solve(new_theory, new_op, eko_filename)
            logger.info(
                "Finished computation of %s - took %f s",
                name,
                time.perf_counter() - start_time,
            )
            if eko_filename.exists():
                rich.print(f"[green]Success:[/] Wrote EKO to {eko_filename}")

    def ekos(self):
        """Compute all ekos."""
        tcard = theory_card.load(self.theory_id)
        self.ekos_path().mkdir(exist_ok=True)
        self.iterate(self.eko, tcard=tcard)

    def fk(self, name, grid_path, tcard, pdfs):
        """Compute a single FK table.

        Parameters
        ----------
        name : str
            grid name, i.e. it's true stem
        grid_path : pathlib.Path
            path to grid
        tcard : dict
            theory card
        pdfs : list[str]
            list of PDF set for comparisons
        """
        # activate logging
        paths = configs.configs["paths"]
        comb_pdf_logs = "-".join(pdfs) if pdfs is not None else "nopdf"
        do_log = self.activate_logging(
            paths["logs"]["fk"], f"{self.theory_id}-{name}-{comb_pdf_logs}.log"
        )

        # Relevant for FONLL-B and FONLL-D: For FFN0 terms, PTO is lower than
        # PTODIS, thus using PTO instead of PTODIS to establish the perturbative
        # order would result in the PTODIS terms that correspond to orders
        # beyond PTO to be neglected
        if "FONLL" in tcard["FNS"] and tcard.get("PTODIS") is not None:
            tcard["PTO"] = tcard["PTODIS"]

        # check if grid contains SV if theory is requesting them (in particular
        # if theory is requesting scheme A or C)
        sv_method = evolve.sv_scheme(tcard)
        xir = tcard["XIR"]
        xif = tcard["XIF"]
        xia = 1.0  # TODO: modify into `tcard["XIA"]`
        # loading grid
        grid = pineappl.grid.Grid.read(grid_path)
        # remove zero subgrid
        grid.optimize()

        # Do you need one or multiple ekos?
        names = get_eko_names(grid_path, name, filter=False)
        eko_filename = [self.ekos_path() / f"{ekoname}.tar" for ekoname in names]
        fk_filename = self.fks_path / f"{name}.{parser.EXT}"
        if fk_filename.exists():
            if not self.overwrite:
                rich.print(f"Skipping existing FK Table {fk_filename}")
                return
        max_as = 1 + int(tcard["PTO"])
        # Check if we are computing FONLL-B fktable and eventually change max_as
        if check.is_fonll_mixed(
            tcard["FNS"],
            grid.channels(),
        ):
            max_as += 1

        # NB: This would not happen for nFONLL
        max_al = 0

        # check if the grid is empty
        if check.is_num_fonll(tcard["FNS"]):
            if (
                pineappl.grid.Order.create_mask(
                    grid.orders(), max_as, max_al, True
                ).size
                == 0
            ):
                rich.print("[green] Skipping empty grid.")
                return

        # check for sv
        if not np.isclose(xir, 1.0):
            check_scvar_evolve(grid, max_as, max_al, check.Scale.REN)
        if sv_method is None:
            if not np.isclose(xif, 1.0):
                check_scvar_evolve(grid, max_as, max_al, check.Scale.FACT)
        # TODO: Add fragmentation scale variations
        # loading ekos to produce a tmp copy
        n_ekos = len(eko_filename)
        with eko.EKO.read(eko_filename[0]) as first_operator:
            # Skip the computation of the fktable if the eko is empty
            if len(first_operator.mu2grid) == 0 and check.is_num_fonll(tcard["FNS"]):
                rich.print("[green] Skipping empty eko for nFONLL.")
                return

            first_eko_tmp_path = (
                first_operator.paths.root.parent
                / f"eko-tmp-{name}-{np.random.rand()}.tar"
            )
            first_operator.deepcopy(first_eko_tmp_path)

        if n_ekos > 1:
            extra_eko_tmp_path = []
            for eko_idx in range(1, n_ekos):
                with eko.EKO.read(eko_filename[eko_idx]) as extra_operators:
                    eko_tmp_path_extra = (
                        first_operator.paths.root.parent
                        / f"eko-tmp-{name}-{np.random.rand()}.tar"
                    )
                    extra_operators.deepcopy(eko_tmp_path_extra)
                    extra_eko_tmp_path.append(eko_tmp_path_extra)

        with eko.EKO.edit(first_eko_tmp_path) as first_operator:
            # Obtain the assumptions hash
            assumptions = theory_card.construct_assumptions(tcard)
            # do it!
            logger.info("Start computation of %s", name)
            logger.info(
                "max_as=%d, max_al=%d, xir=%f, xif=%f, xia=%f",
                max_as,
                max_al,
                xir,
                xif,
                xia,
            )
            start_time = time.perf_counter()
            rich.print(
                rich.panel.Panel.fit(
                    "Computing ...", style="magenta", box=rich.box.SQUARE
                ),
                f"   {grid_path}\n",
                f"+ {eko_filename}\n",
                f"= {fk_filename}\n",
                f"with max_as={max_as}, max_al={max_al}, xir={xir}, xif={xif}, xia={xia}",
            )

            operators = [first_operator]
            if n_ekos > 1:
                for eko_idx in range(len(extra_eko_tmp_path)):
                    operators.append(eko.EKO.edit(extra_eko_tmp_path[eko_idx]))

            _grid, _fk, comparison = evolve.evolve_grid(
                grid,
                operators,
                fk_filename,
                max_as,
                max_al,
                xir=xir,
                xif=xif,
                xia=xia,
                assumptions=assumptions,
                comparison_pdfs=pdfs,
                meta_data={"theory_card": json.dumps(tcard)},
            )

            if n_ekos > 1:
                # Remove tmp ekos
                for eko_tmp in extra_eko_tmp_path:
                    eko_tmp.unlink()

        # Remove tmp ekos
        first_eko_tmp_path.unlink()

        logger.info(
            "Finished computation of %s - took %f s",
            name,
            time.perf_counter() - start_time,
        )
        if do_log and comparison is not None:
            logger.info(
                f"Comparison with PDFs: {comb_pdf_logs}: \n {comparison.to_string()}"
            )
        if fk_filename.exists():
            rich.print(f"[green]Success:[/] Wrote FK table to {fk_filename}")

    def fks(self, pdfs):
        """Compute all FK tables.

        Parameters
        ----------
        pdfs : list(str)
            list of PDF sets to be used for the comparisons
        """
        tcard = theory_card.load(self.theory_id)
        self.fks_path.mkdir(exist_ok=True)
        self.iterate(self.fk, tcard=tcard, pdfs=pdfs)

    def construct_ren_sv_grids(self, flavors):
        """Construct renormalization scale variations terms for all the grids in a dataset."""
        tcard = theory_card.load(self.theory_id)
        self.iterate(self.construct_ren_sv_grid, tcard=tcard, flavors=flavors)

    def construct_ren_sv_grid(self, name, grid_path, tcard, flavors):
        """Construct renormalization scale variations terms for a grid."""
        max_as = int(tcard["PTO"])
        rich.print(f"Computing renormalization scale variations for {name}")
        scale_variations.compute_ren_sv_grid(grid_path, max_as, flavors)
