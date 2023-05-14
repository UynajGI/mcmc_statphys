"""Console script for mcmc_statphys."""
import sys

import click


@click.command()
def main(args=None):
    """Console script for mcmc_statphys."""
    click.echo("Replace this message by putting your code into "
               "mcmc_statphys.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
