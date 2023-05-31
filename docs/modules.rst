Module
======

In the lastest version(0.3.0) of ``mcmc_statphys``, we have these modules:

``mcmc_statphys.model``
-----------------------

In this subpackage, we have Ising model, Heisenberg model, Potts model, and XY model.

``Ising``
~~~~~~~~~

The Ising model is a mathematical model of ferromagnetism in statistical mechanics. The model consists of discrete variables that represent magnetic dipole moments of atomic spins that can be in one of two states (+1 or −1). The spins are arranged in a graph, usually a lattice, allowing each spin to interact with its neighbors. The model allows the identification of phase transitions, as a simplified model of reality.

The detial of the Ising model can be found in `Wikipedia <https://en.wikipedia.org/wiki/Ising_model>`__.

-  attribute

   -  L: ``int``, the length of the lattice
   -  dim: ``int``, the dimension of the lattice
   -  J: ``float``, the interaction strength
   -  H: ``float``, the external field strength
   -  spin: ``numpy.ndarray``, the spin configuration
   -  energy: ``float``, the energy of the spin configuration
   -  magnetization: ``float``, the magnetization of the spin
      configuration
   -  type: ``str``, the type of the model

-  method

   -  ``__init__(self, L, dim, J, H)``: initialize the Ising model
   -  ``set_spin(self, spin)``: set the spin configuration Args: spin:
      ``numpy.ndarray``, the spin configuration
   -  ``get_energy(self)``: get the energy of the spin configuration
      Returns: energy\ ``float``: The total energy of the system
   -  ``get_magnetization(self)``: get the magnetization of the spin
      configuration Returns: magnetization\ ``float``: The total
      magnetization of the system

``Heisenberg``
~~~~~~~~~~~~~~

The Heisenberg model is a mathematical model in quantum mechanics used to explain the magnetic properties of materials. It arises from the exchange interaction between the spins of electrons. The spins in a Heisenberg model are treated as classical vectors. The spins are typically taken to be three-component vectors (x, y, z), but other representations are possible. The Heisenberg model is a type of an Ising model, namely, a model of interacting spins. 

The detial of the Heisenberg model can be found in `Wikipedia <https://en.wikipedia.org/wiki/Classical_Heisenberg_model>`__.

Attribute and method are the same as ``Ising``.

``Potts``
~~~~~~~~~

The Potts model, a generalization of the Ising model, is a model in statistical mechanics that describes a collection of spins that can take on q different states on a regular lattice. It was introduced by the English mathematician R. B. Potts in 1952. The states of the spins are denoted s = 1, …, q. 

The detial of the Potts model can be found in `Wikipedia <https://en.wikipedia.org/wiki/Potts_model>`__.

Attribute and method are the same as ``Ising``.

``XY``
~~~~~~

The XY model is a model of a magnet where the spins lie on a plane and interact with each other only through their orientation. It is named after the two-dimensional XY model, but the same name is also used to describe related models in different numbers of dimensions. The XY model  is one of the simplest statistical models exhibiting a phase transition with a continuous symmetry group.

The detial of the XY model can be found in `Wikipedia <https://en.wikipedia.org/wiki/Classical_XY_model>`__.

Attribute and method are the same as ``Ising``.

``mcmc_statphys.algorithm``
---------------------------

In this subpackage, we have Metropolis algorithm, Wolff algorithm, and Annelaing algorithm.

``Metropolis``
~~~~~~~~~~~~~~

The Metropolis algorithm, also referred to as the Metropolis Monte Carlo algorithm, is a Monte Carlo method used to generate an approximate probability distribution through a sequence of random samples. The Metropolis algorithm is often used in statistical physics to compute the canonical ensemble and in chemistry to simulate the behavior of atoms in a material.

The detial of the Metropolis algorithm can be found in `Wikipedia <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>`__.

-  attribute

   -  model: ``mcmc_statphys.model``, the model to simulate
   -  name: ``str``, the name of the algorithm
   -  data: ``dict``, the data of the iteration(with the unique
      ``uid`` for each iteration)
   -  parameter: ``str``, the name of the parameter

-  method

   -  ``iter_sample(self, T: float, uid: str = None) -> str``: 
      
      Do one iteration of the algorithm
      
      Args: 

      - T: ``float``, the temperature
      - uid: ``str``, the unique id of the iteration
      
      Returns:

      - uid: ``str``, the unique id of the iteration
   
   -  ``equil_sample(self, T: float, max_iter: int = 1000, uid: str = None) -> str``:
      
      Do the equilibration of the algorithm 
      
      Args: 

      - T: ``float``, the temperature of sample 
      - max_iter: ``int``, the maximum number of iterations (Default: 1000) 
      - uid: ``str``, the unique id of the iteration
      
      Returns:

      - uid: ``str``, the unique id of the iteration

   -  ``param_sample(self, max_iter: int = 1000) -> Dict``: 

      Do the parameter sampling of the algorithm 

      Args:

      - max_iter: ``int``, the maximum number of iterations (Default: 1000) 

      Returns: 
      
      - param_data: ``dict``, the data of the parameter sampling

``Wolff``
~~~~~~~~~

The Wolff algorithm is a Monte Carlo algorithm for generating configurations of an Ising model or Potts model. It was invented by Ulli Wolff in 1989.

The detial of the Wolff algorithm can be found in `Wikipedia <https://en.wikipedia.org/wiki/Wolff_algorithm>`__.

Attribute and method are the same as ``Metropolis``.

``Anneal``
~~~~~~~~~~

Simulated annealing is a probabilistic technique for approximating the global optimum of a given function. Specifically, it is a metaheuristic to approximate global optimization in a large search space for an optimization problem. It is often used when the search space is discrete (e.g., all tours that visit a given set of cities). For problems where finding an approximate global optimum is more important than finding a precise local optimum in a fixed amount of time, simulated annealing may be preferable to alternatives such as gradient descent.

The detial of the Annelaing algorithm can be found in `Wikipedia <https://en.wikipedia.org/wiki/Simulated_annealing>`__.

Attribute and method are the same as ``Metropolis``, except of ``equil_sample``.

-  method
   - ``equil_sample(self, targetT: float, max_iter: int = 1000, highT=None, dencyT=0.9, uid: str = None, ) -> str``:

      Do the equilibration of the algorithm 
      
      Args: 
      
      - targetT: ``float``, the target temperature of sample 
      - max_iter: ``int``, the maximum number of iterations (Default: 1000) 
      - highT: ``float``, the high temperature of the annealing (Default: 2 \* targetT) 
      - dencyT: ``float``, the density of the annealing (Default: 0.9) 
      - uid: ``str``, the unique id of the iteration 
      
      Returns: 
      
      - uid: ``str``, the unique id of the iteration


In ``algorithm`` subpackage, we also have same analysis method

-  ``mean(uid: str, column: str) -> float``:

   Calculate the mean of the data
   
   Args: 
   
   - uid: ``str``, the unique id of the iteration
   - column: ``str``, the name of the column 

   Returns: 
   
   - mean: ``float``, the mean of the data

-  ``std(uid: str, column: str) -> float``: 
   
   Calculate the standard deviation of the data
   
   Args: 
   
   - uid: ``str``, the unique id of the iteration
   - column: ``str``, the name of the column 

   Returns: 
   
   - std: ``float``, the standard deviation of the data

-  ``var(uid: str, column: str) -> float``: 
   
   Calculate the variance of the data
   
   Args: 
   
   - uid: ``str``, the unique id of the iteration
   - column: ``str``, the name of the column 

   Returns: 
   
   - var: ``float``, the variance of the data

- ``norm(uid: str, column: str) -> float``: 
   
   Calculate the norm of the data
   
   Args: 
   - uid: ``str``, the unique id of the iteration
   - column: ``str``, the name of the column 

   Returns: 
   
   - norm: ``float``, the norm of the data

-  ``cv(uid: str, column: str) -> float``: 

   Calculate the coefficient of variation of the data 
   
   Args: 
   - uid: ``str``, the unique id of the iteration 
   - column: ``str``, the name of the column

   Returns:
   
   - cv: ``float``, the coefficient of variation

-  ``diff(uid: str, column: str) -> float``: 
   
   Calculate the difference of the data
   
   Args:
   
   - uid: ``str``, the unique id of the iteration column: ``str``, the name of the column 
   
   Returns: 
   
   - diff: ``float``, the difference of the data

-  ``getcolumn(uid: str, column: str) -> numpy.ndarray``:

   Get the column of the data
   
   Args:
   - uid: ``str``, the unique id of the iteration 
   - column: ``str``, the name of the column 

   Returns:
   
   - column: ``numpy.ndarray``, the column of the data

-  ``svd(self, uid: str or Dict or List[str], norm: bool = True) -> np.array:``: 
   
   Calculate the singular value decomposition of the data
   
   Args: 
   
   - uid: ``str`` or ``Dict`` or ``List[str]``, the unique id of the iteration
   - norm: ``bool``, whether to normalize the data (Default: True)

   Returns: 
   
   - svd: ``numpy.ndarray``, the singular value decomposition of the data

``mcmc_statphys.draw``
----------------------

In this subpackage, we have the draw of the data.
