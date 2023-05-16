"""Console script for mcmc_statphys."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("python-mc-stat-phys")
    click.echo("=" * len("python-mc-stat-phys"))
    click.echo("A library project of Monte Carlo simulation algorithms for some statistical physics models (in particular, the Ising model and its variants).")


if __name__ == "__main__":
    main()  # pragma: no cover
