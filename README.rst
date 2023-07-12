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
    >>> model = mcsp.model.Ising(12) # 12x12 Ising model
    >>> algorithm = mcsp.algorithm.Metropolis(model) # Metropolis algorithm
    >>> uid = algorithm.equil_sample(T=1, max_iter=1000) # sample until equilibrium
    >>> fig = mcsp.draw.Plot(algorithm)
    >>> fig.curve(uid=uid, comlumn='energy') # plot the energy curve

Install
-------

the latest version of mcmc_statphys:

.. code-block:: console

    $ pip install mcmc_statphys

upgrade to the latest version:

.. code-block:: console

    $ pip install --upgrade mcmc_statphys

Bugs
----

Please report any bugs that you find `here`_. Or, even better, fork the repository on `GitHub`_ and create a pull request (PR). We welcome all changes, big or small, and we will help you make the PR if you are new to git (just ask on the issue and/or see `CONTRIBUTING`_).

.. _here: https://github.com/uynajgi/mcmc_statphys/issues
.. _GitHub: https://github.com/uynajgi/mcmc_statphys/
.. _CONTRIBUTING: https://mcmc-statphys.readthedocs.io/en/latest/contributing.html

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
