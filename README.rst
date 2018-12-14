pytsmp
======

.. image:: https://gitlab.com/mcnuggets23/pytsmp/badges/master/pipeline.svg
       :target: https://gitlab.com/mcnuggets23/pytsmp/commits/master
       :alt: Latest build status

.. image:: https://gitlab.com/mcnuggets23/pytsmp/badges/master/coverage.svg
        :target: https://gitlab.com/mcnuggets23/pytsmp/commits/master
        :alt: Latest coverage status

.. comment
    .. image:: https://img.shields.io/pypi/v/pytsmp.svg
        :target: https://pypi.python.org/pypi/pytsmp
        :alt: Latest PyPI version

.. comment
    .. image:: https://codecov.io/gh/kithomak/pytsmp/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/kithomak/pytsmp/branch/master
        :alt: Latest Codecov status

.. comment
    .. image:: https://travis-ci.org/kithomak/pytsmp.png
       :target: https://travis-ci.org/kithomak/pytsmp
       :alt: Latest Travis CI build status

``pytsmp`` is a Python implementation of the matrix profile. More details about matrix profile can be
found in the `UCR Matrix Profile Page <http://www.cs.ucr.edu/~eamonn/MatrixProfile.html>`_
by the paper authors.

Currently, only MASS and STAMP (single-core, no GPU support) is implemented.
I plan to implement other algorithms (STOMP, SCRIMP++), their parallelized version,
as well as convenience functions such as motif finding and anomaly detection.

The original implementation (in R) of the paper authors from the UCR group can be found
`here <https://github.com/franzbischoff/tsmp>`_.


Installation
------------

``pytsmp`` is available via pip.

.. code:: bash

   pip install pytsmp


Usage
-----

To compute the matrix profile using STAMP, use the following code.

.. code:: python

   import numpy as np
   from pytsmp import STAMP

   # create a 1000 step random walk and a random query
   ts = np.cumsum(np.random.randint(2, size=(1000,)) * 2 - 1)
   query = np.random.rand(200)

   # Create the STAMP object. Note that computation starts immediately.
   mp = STAMP(ts, query, window_size=50)  # window_size must be specified as a named argument

   # get the matrix profile and the profile indexes
   mat_profile, ind_profile = mp.get_profiles()

Incremental of the time series and the query is supported.

.. code:: python

   import numpy as np
   from pytsmp import STAMP

   # create a 1000 step random walk and its matrix profile
   ts = np.cumsum(np.random.randint(2, size=(1000,)) * 2 - 1)
   mp = STAMP(ts, window_size=50)
   mat_profile, _ = mp.get_profiles()

   # create the matrix profile of the first 999 steps and increment the last step later
   mp_inc = STAMP(ts[:-1], window_size=50)
   mp_inc.update_ts1(ts[-1])  # similarly, you can update the query by update_ts2()
   mat_profile_inc, _ = mp_inc.get_profiles()

   print(np.allclose(mat_profile, mat_profile_inc))  # True


Benchmark
---------

Perform a simple trial run on a random walk with 40000 data points.

.. code:: python

   import numpy as np
   from pytsmp import STAMP

   ts = np.cumsum(np.random.randint(2, size=(40000,)) * 2 - 1)

   # ipython magic command
   %timeit mp = STAMP(ts, window_size=1000, verbose=False)

   # and similarly for STOMP and SCRIMP

On my MacBook Pro with 2.2 GHz Intel Core i7, the results are (all over 7 runs, 1 loop each):

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1
   :align: center

   * - Algorithm
     - Data Size
     - Window Size
     - Elapsed Time
   * - STAMP
     - 40000
     - 1000
     - 2min 14s ± 392ms
   * - STOMP
     - 40000
     - 1000
     - 22.1s ± 52.8ms
   * - SCRIMP
     - 40000
     - 1000
     - 23.6s ± 402ms



.. comment
   License
   -------


Reference
---------

C.C.M. Yeh, Y. Zhu, L. Ulanova, N. Begum, Y. Ding, H.A. Dau, D. Silva, A. Mueen and E. Keogh.
"Matrix profile I: All pairs similarity joins for time series: A unifying view that includes
motifs, discords and shapelets". IEEE ICDM 2016.

Y. Zhu, Z. Zimmerman, N.S. Senobari, C.C.M. Yeh, G. Funning, A. Mueen, P. Berisk and E. Keogh.
"Matrix Profile II: Exploiting a Novel Algorithm and GPUs to Break the One Hundred Million
Barrier for Time Series Motifs and Joins". IEEE ICDM 2016.

Y. Zhu, C.C.M. Yeh, Z. Zimmerman, K. Kamgar and E. Keogh.
"Matrix Proﬁle XI: SCRIMP++: Time Series Motif Discovery at Interactive Speed". IEEE ICDM 2018.


.. comment
   `pytsmp` was written by Kit-Ho Mak at `ASTRI <https://www.astri.org>`_.


