=======
History
=======

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[Unreleased]
------------

[0.4.0] - 2023-05-20
--------------------

Added
~~~~~

* Add `imshow` in `draw`
* Add `animate` in `draw`

Fixed
~~~~~

* Fix: `Wolff` and `Anneal` uid does not return

[0.3.0] - 2023-05-18
--------------------

Added
~~~~~

* Add `cv` in `analysis` to calculate the cv
* Add `spin2svd` in `analysis`
* Add `uid2svd` in `analysis`

[0.2.1] - 2023-05-17
--------------------

Fixed
~~~~~

* Fix the bug of saving `_get_per_magnetization` in `_save_data` in `algorithm`

Added
~~~~~

* Add moudle `analysis` to analyze the data
* Add moudle `draw` to draw the figures
* Add method `setspin` in `model`
* Add tqmd to show the progress bar

Doc
~~~

* Add documentation to README
* Add documentation to Usages

Changed
~~~~~~~

* Change the methods in the `analysis` module: removed the `Sample` and `ParameterSample` classes, added `Metropolis`, `Wolff`, `Anneal` classes and several methods

[0.1.2] - 2023-05-15
--------------------

Security
~~~~~~~~

* Add function annotations to all functions
* Add type hints to all functions
* Add type hints to all variables
* Change `mcmc_statphys.py` to `method.py`

Doc
~~~

* Add documentation to README

[0.1.1] - 2023-05-14
--------------------

* First release on PyPI.
