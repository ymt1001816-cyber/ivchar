"""Command-line entry point for ivchar."""

import argparse

from ivchar import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ivchar", description=__doc__)
    parser.add_argument("--version", action="version", version=f"ivchar {__version__}")
    parser.parse_args(argv)
    parser.print_help()


if __name__ == "__main__":
    main()
