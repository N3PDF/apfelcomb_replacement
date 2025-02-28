"""CLI entry point to generation of the inclusion of kfactor in a grid."""

import pathlib

import rich_click as click

from .. import kfactor
from ._base import command, config_option, load_config


@command.command("kfactor")
@config_option
@click.argument("theoryID", type=int)
@click.argument("dataset", type=str)
@click.argument("kfactor_folder", type=click.Path(exists=True))
@click.argument("target_folder", type=click.Path(exists=True))
@click.argument("pto_to_update", type=int)
@click.option("--order_exists", is_flag=True, help="Overwrite an existing order.")
def kfactor_inclusion(
    cfg,
    theoryid,
    dataset,
    kfactor_folder,
    target_folder,
    pto_to_update,
    order_exists,
):
    """Construct new grid with kfactor included."""
    load_config(cfg)
    kfactor_folder = pathlib.Path(kfactor_folder)
    target_folder = pathlib.Path(target_folder)
    kfactor.apply_to_dataset(
        theoryid,
        dataset,
        kfactor_folder,
        pto_to_update,
        target_folder=target_folder,
        order_exists=order_exists,
    )
