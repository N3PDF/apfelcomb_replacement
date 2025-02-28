"""Adds global CLI options."""

import pathlib
import sys

import rich_click as click

from .. import configs

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def help_requested():
    """Check if you are requesting help."""
    return len(set(CONTEXT_SETTINGS["help_option_names"]) & set(sys.argv)) > 0


@click.group(context_settings=CONTEXT_SETTINGS)
def command():
    """pineko: Combine PineAPPL grids and EKOs into FK tables."""


config_option = click.option(
    "-c",
    "--configs",
    "cfg",
    default=None,
    type=click.Path(resolve_path=True, path_type=pathlib.Path),
    help="Explicitly specify config file (it has to be a valid TOML file).",
)


def load_config(cfg):
    """Load configuration files."""
    # if only help is needed, return before loading
    if help_requested():
        return
    path = configs.detect(cfg)
    base_configs = configs.load(path)
    configs.configs = configs.defaults(base_configs)
    if cfg is not None:
        print(f"Configurations loaded from '{path}'")
