# Hypertuning

QumFu (QUell My FUnction) is a package to perform function optimization in Python via a wide range
of state-of-the-art algorithms, mainly focused on machine learning hyper-parameter tuning through the library built-in
wrappers around Scikit-Learn, RAPIDS, and Tensorflow.

It provides:
- Tools to set the search space allowing conditional spaces as well (sampling parameters based on the already sampled
    values of other parameters).
- Different range of single- and multi-objective optimization algorithms.
- Evaluation metrics for single- and multi-objective problems.

Besides the main focus of the library to be used in the task of machine learning hyper-parameters tuning, the library
functions can be applied to any optimization problem in Python.

It is based in the following modules:

* **search_space**: contains all functions to define the search space of the input variables. The main search space is
  defined as an object which contains the distributions from where to draw each input value. It allows to define
  conditional input values based on the value drawn for others. Those distributions can be defined as:
    * ```integer_random```: it draws a random integer between the given upper and lower bounds. It can be selected if 
      each bound is inclusive or exclusive, and also if a posterior function has to be applied to the integer draw.
        
    * ```integer_normal```: it draws an integer from a Gaussian distribution given its mean and standard deviation. It 
      is if a posterior function has to be applied to the integer draw.
      
    * ```float_random```: same than integer_random, but drawing a float value.
      
    * ```float_random```: same than integer_normal, but drawing a float value.
    
    * ```choice```: it draws randomly one of the given options. Those could be any type.
  
    * ```static```: placeholder to store constants that do not need to be optimized.
    
    
    

* **tracking_metrics**: contains all functions to define the goal function, being this a single value or multivariate.


* **optimization**: contains all algorithms to perform function optimization. So far the following techniques has been
    implemented:
  * ```random```: it searches for a minimum on the tracking metrics based on the defined search space, where each run
    contains random sampled values. It checks that a run is not repeating an already studied input space. Also allows
    the user to specify a minimum distance between searched points.
    
  * ```bayesian```: it searched for a minimum on the tracking metrics based on a Bayesian search. After the first drawn
    of input space and computation of the tracking metrics, it computes an Ensemble Gaussian Distributions and selects
    as next input space the predicted optimal or another random set of values based on the explorer (delta) and greedy
    (gamma) parameters, where *1 = delta + gamma*. The former indicates how the model have to tend to explore new 
    spaces, while the latter is related to how the model tends to explore the predicted best values from the Gaussian 
    distributions. Those parameters are updated along the run based on the ro parameter, which indicates how fast the 
    explorer behaviour decays into greedy.
    

  * ```rl```: it searches for a minimum on the tracking metrics based on a Q-Learning reinforcement learning algorithm.
    The behavious of this algorithm is also controller with the parameters delta, gamma, and ro.
    

**All QumFu wheels distributed on PyPI are GPLv3 licensed**
