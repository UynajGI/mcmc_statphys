Usages
======

``mcmc_statphys`` is a python package for Monte Carlo simulations of statistical physics models. It is designed to be easy to use and easy to extend. It is also designed to be easy to use in a Jupyter notebook.

To use mcmc_statphys in a project

.. code:: python

   import mcmc_statphys as mcpy

Quick Start
-----------

Choose a model
~~~~~~~~~~~~~~

First, create a model object that you want to simulate. For example, to create a 2D Ising model with 10x10 spins (Default: H=0, J=1)

*all results mey be different, it’s normal.*

.. code:: python

   from mcmc_statphys.model import Ising

   model = Ising(L=10, dim=2)

Now you can get the ``energy``/``magnetization``/``spin`` of the model:

.. code:: python

   model.energy
   # 20.0
   model.magnetization
   # 12
   model.spin
   # array([[-1,  1,  1, -1,  1,  1,  1,  1,  1,  1],
   #        [ 1, -1, -1,  1, -1, -1, -1, -1, -1,  1],
   #        [-1,  1,  1, -1, -1,  1,  1,  1,  1, -1],
   #        [ 1, -1,  1, -1,  1,  1, -1,  1,  1, -1],
   #        [-1,  1, -1,  1,  1,  1,  1,  1, -1,  1],
   #        [-1,  1,  1,  1,  1, -1,  1, -1, -1,  1],
   #        [ 1, -1,  1, -1,  1, -1, -1, -1, -1, -1],
   #        [ 1,  1, -1, -1, -1,  1,  1, -1,  1, -1],
   #        [-1,  1, -1,  1,  1,  1,  1,  1,  1, -1],
   #        [ 1,  1, -1, -1,  1, -1, -1,  1,  1, -1]], dtype=int8)

Choose a algorithm
~~~~~~~~~~~~~~~~~~

Next, create a algorithm object that you want to use. For example, to create a Metropolis algorithm with 1000 steps

.. code:: python

   from mcmc_statphys.algorithm import Metropolis
   algo = Metropolis(model=model)

Now you can choose a sample method and get the result, e.g. ``equil_sample``:

.. code:: python

   uid = algo.equil_sample(T=1.0, max_iter=1000) # set the sample temperature and max iteration
   # progress bar
   # 100%|██████████| 1000/1000 [00:09<00:00, 101.74it/s]
   uid
   '607ecc20f46d11ed948300e04c6807cc' # your uid must be different, because it is unique

Notice that the ``uid`` is the unique id of the sample. In this package, we use ``uid`` to identify a sample process. We save the data of the sample in a attribute called ``data`` of the algorithm object, which type is a ``pd.DataFrame``. You can get the data of the sample by using ``data.loc[uid]``:

.. code:: python

   algo.data.loc[uid]

+===+=============+============+============+============+============+
| i | T           | H          | energy     | mag        | spin       |
| t |             |            |            | netization |            |
| e |             |            |            |            |            |
| r |             |            |            |            |            |
+===+=============+============+============+============+============+
| 1 | 1.0         | 0          | 20.0       | 10        | …          |
+---+-------------+------------+------------+------------+------------+
| 2 | 1.0         | 0          | 20.0       | 10         | …          |
+---+-------------+------------+------------+------------+------------+
| 3 | 1.0         | 0          | 20.0       | 8          | …          |
+---+-------------+------------+------------+------------+------------+
| 4 | 1.0         | 0          | 20.0       | 8          | …          |
+---+-------------+------------+------------+------------+------------+
| 5 | 1.0         | 0          | 20.0       | 10         | …          |
+---+-------------+------------+------------+------------+------------+

Details of how to use the ``pandas`` can be found in the `User
Guide <https://pandas.pydata.org/docs/user_guide/index.html>`__

The uid mean that you can continue the sample process by using the same uid. For example, you can continue the sample process by using the same uid:

.. code:: python

   from uuid import uuid1
   uid1 = uuid1().hex
   uid2 = uuid1().hex
   algo2 = Metropolis(model=model)
   algo2.iter_sample(T=1.0, uid=uid1)
   algo2.iter_sample(T=1.0, uid=uid2)
   algo2.data

+===+===+=============+============+============+============+============+
| u | i | T           | H          | energy     | mag        | spin       |
| i | t |             |            |            | netization |            |
| d | e |             |            |            |            |            |
|   | r |             |            |            |            |            |
+===+===+=============+============+============+============+============+
| u | i | T           | H          | energy     | mag        | spin       |
| i | t |             |            |            | netization |            |
| d | e |             |            |            |            |            |
|   | r |             |            |            |            |            |
+---+---+-------------+------------+------------+------------+------------+
| f | 1 | 1.0         | 0          | 20.0       | 10         | …          |
| 3 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| 9 |   |             |            |            |            |            |
| 7 |   |             |            |            |            |            |
| 2 |   |             |            |            |            |            |
| 9 |   |             |            |            |            |            |
| 9 |   |             |            |            |            |            |
| f |   |             |            |            |            |            |
| 4 |   |             |            |            |            |            |
| 7 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 1 |   |             |            |            |            |            |
| 1 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| d |   |             |            |            |            |            |
| 9 |   |             |            |            |            |            |
| 2 |   |             |            |            |            |            |
| b |   |             |            |            |            |            |
| f |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 4 |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
| 6 |   |             |            |            |            |            |
| 8 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 7 |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
+---+---+-------------+------------+------------+------------+------------+
| f | 1 | 1.0         | 0          | 20.0       | 10         | …          |
| 3 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| a |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| 2 |   |             |            |            |            |            |
| 8 |   |             |            |            |            |            |
| b |   |             |            |            |            |            |
| f |   |             |            |            |            |            |
| 4 |   |             |            |            |            |            |
| 7 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 1 |   |             |            |            |            |            |
| 1 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| d |   |             |            |            |            |            |
| 9 |   |             |            |            |            |            |
| b |   |             |            |            |            |            |
| 8 |   |             |            |            |            |            |
| 4 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 4 |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
| 6 |   |             |            |            |            |            |
| 8 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 7 |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
+---+---+-------------+------------+------------+------------+------------+

Continue the sample process by using the same uid:

.. code:: python

   algo2.iter_sample(T=1.0, uid=uid2)
   algo2.data

+===+===+=============+============+============+============+============+
| u | i | T           | H          | energy     | mag        | spin       |
| i | t |             |            |            | netization |            |
| d | e |             |            |            |            |            |
|   | r |             |            |            |            |            |
+===+===+=============+============+============+============+============+
| u | i | T           | H          | energy     | mag        | spin       |
| i | t |             |            |            | netization |            |
| d | e |             |            |            |            |            |
|   | r |             |            |            |            |            |
+---+---+-------------+------------+------------+------------+------------+
| f | 1 | 1.0         | 0          | 20.0       | 10         | …          |
| 3 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| 9 |   |             |            |            |            |            |
| 7 |   |             |            |            |            |            |
| 2 |   |             |            |            |            |            |
| 9 |   |             |            |            |            |            |
| 9 |   |             |            |            |            |            |
| f |   |             |            |            |            |            |
| 4 |   |             |            |            |            |            |
| 7 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 1 |   |             |            |            |            |            |
| 1 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| d |   |             |            |            |            |            |
| 9 |   |             |            |            |            |            |
| 2 |   |             |            |            |            |            |
| b |   |             |            |            |            |            |
| f |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 4 |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
| 6 |   |             |            |            |            |            |
| 8 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 7 |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
+---+---+-------------+------------+------------+------------+------------+
| f | 1 | 1.0         | 0          | 20.0       | 10         | …          |
| 3 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| a |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| 2 |   |             |            |            |            |            |
| 8 |   |             |            |            |            |            |
| b |   |             |            |            |            |            |
| f |   |             |            |            |            |            |
| 4 |   |             |            |            |            |            |
| 7 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 1 |   |             |            |            |            |            |
| 1 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| d |   |             |            |            |            |            |
| 9 |   |             |            |            |            |            |
| b |   |             |            |            |            |            |
| 8 |   |             |            |            |            |            |
| 4 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| e |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 4 |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
| 6 |   |             |            |            |            |            |
| 8 |   |             |            |            |            |            |
| 0 |   |             |            |            |            |            |
| 7 |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
| c |   |             |            |            |            |            |
+---+---+-------------+------------+------------+------------+------------+
|   | 2 | 1.0         | 0          | 20.0       | 8          | …          |
+---+---+-------------+------------+------------+------------+------------+

Analyze the data
~~~~~~~~~~~~~~~~

If you want to analyze the data, you can use the ``analyze`` module. For example, to get the energy distribution of the sample:

.. code:: python

   uid3 = algo.equil_sample(T=1.0, max_iter=1000)
   energy_mean = algo.mean(uid=uid3,column='energy') 
   energy_mean
   # -124.172
   # e.t.c.

Plot the data
~~~~~~~~~~~~~

If you want to plot the data, you can use the ``draw`` module. For example, to plot the energy distribution of the sample:

.. code:: python

   from mcmc_statphys import draw
   uid4 = algo.equil_sample(T=1.0, max_iter=1000)
   fig = draw.Plot(algo)
   fig.curve(uid=uid4, column='energy')


Animate the data

If you want to animate the data, you can use the ``animate`` module. For example, to animate the spin of the sample:

.. code:: python

   from mcmc_statphys import draw
   uid5 = algo.equil_sample(T=1.0, max_iter=1000)
   ani = animate.Animation(algo)
   ani.animate(uid=uid5) # the animation will be saved in the ./uid folder