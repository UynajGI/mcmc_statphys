=============
mcmc_statphys
=============


.. image:: https://img.shields.io/pypi/v/mcmc_statphys.svg
        :target: https://pypi.python.org/pypi/mcmc_statphys

.. image:: https://img.shields.io/travis/uynajgi/mcmc_statphys.svg
        :target: https://travis-ci.com/uynajgi/mcmc_statphys

.. image:: https://readthedocs.org/projects/mcmc-statphys/badge/?version=latest
        :target: https://mcmc-statphys.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




A Python package of Monte Carlo simulation algorithms for some statistical physics models (in particular, the Ising model and its variants).


* Free software: MIT license
* Documentation: https://mcmc-statphys.readthedocs.io.


Features
--------

A simple Ising model simulation.

.. code-block:: python

    >>> import mcmc_statphys as mcsp
    >>> import matplotlib.pyplot as plt # We have not yet produced a built-in plotting module. Stay tuned.
    >>> model = mcsp.model.Ising(16)
    >>> simulator = mcsp.method.Simulation(model)
    >>> energy, magnetization, _ = simulator.metropolis_sample(T=1, max_iter=10000)
    >>> plt.plot(energy)

Install the latest version of mcmc_statphys: 

.. code-block:: console

    $ pip install mcmc_statphys

Bugs
----

Please report any bugs that you find `here`_. Or, even better, fork the repository on `GitHub` and create a pull request (PR). We welcome all changes, big or small, and we will help you make the PR if you are new to git (just ask on the issue and/or see `CONTRIBUTING`).

.. _here: https://github.com/uynajgi/mcmc_statphys/issues
.. _GitHub: https://github.com/uynajgi/mcmc_statphys/
.. _CONTRIBUTING: https://mcmc-statphys.readthedocs.io/en/latest/contributing.html

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
