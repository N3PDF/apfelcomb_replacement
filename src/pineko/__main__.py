import argparse

import pineappl
import eko
import rich

from . import convolute, check_grid_and_eko_compatible
from .comparator import compare

ap = argparse.ArgumentParser()
cmds = ap.add_subparsers()

# convolute
def cli_convolute(args):
    return convolute(args.pineappl, args.eko, args.fktable, args.pdf)

conv = cmds.add_parser("convolute")
conv.add_argument("pineappl", help="Path to PineAPPL grid")
conv.add_argument("eko", help="Path to EKO output")
conv.add_argument("fktable", help="Target path for FK table")
conv.add_argument("--pdf", default=None, help="if given, print comparison table")
conv.set_defaults(func=cli_convolute)

# check
def cli_check(args):
    pineappl_grid = pineappl.grid.Grid.read(args.pineappl)
    operators = eko.output.Output.load_yaml_from_file(args.eko)
    try:
        check_grid_and_eko_compatible(pineappl_grid, operators)
        rich.print("[green]Success:[/] grids are compatible")
    except ValueError as e:
        rich.print("[red]Error:[/]",e)

conv = cmds.add_parser("check")
conv.add_argument("pineappl", help="Path to PineAPPL grid")
conv.add_argument("eko", help="Path to EKO output")
conv.set_defaults(func=cli_check)

# compare
def cli_compare(args):
    print(compare(args.pineappl, args.fktable, args.pdf))

conv = cmds.add_parser("compare")
conv.add_argument("pineappl", help="Path to PineAPPL grid")
conv.add_argument("fktable", help="Path to FK table grid")
conv.add_argument("pdf", help="PDF set name")
conv.set_defaults(func=cli_compare)

# doit
args = ap.parse_args()
try:
    args.func(args)
except AttributeError:
    ap.print_help()
