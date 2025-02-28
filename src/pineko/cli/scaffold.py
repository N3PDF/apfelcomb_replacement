"""'scaffold' mode of CLI."""

import rich

from .. import configs, scaffold
from ._base import command, config_option, load_config


@command.group("scaffold")
@config_option
def scaffold_(cfg):
    """Detect amd load configuration file."""
    load_config(cfg)


@scaffold_.command()
def new():
    """Create all the folders to set up a new project."""
    scaffold.set_up_project(configs.configs)


@scaffold_.command()
def check():
    """Check if all the configurations are correct."""
    check_res = scaffold.check_folders(configs.configs)
    if check_res.success:
        rich.print("[green]Success:[/] All the folders are correctly configured.")
    else:
        rich.print("[red]Error:[/] Project is not correctly configured.")
        for conf in check_res.confs:
            rich.print(f"[red]Missing entry in conf file: '{conf}'")
        for folder in check_res.folders.values():
            if not isinstance(folder, dict):
                rich.print(f"[red]Missing folder:\n{folder}")
            else:
                for log_key in folder:
                    rich.print(f"[red]Missing folder: \n{folder[log_key]}")
