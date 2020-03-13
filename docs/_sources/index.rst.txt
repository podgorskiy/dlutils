.. _topics-index:

=================================================
Welcome to dlutils |version|  documentation!
=================================================

.. toctree::
   :caption: First steps
   :maxdepth: 5
   :hidden:

   readme
   batch_provider
   checkpointer
   run
   download
   shuffle
   timer
   epoch
   measures
   cache
   async
   reader

:doc:`readme`
    About.

:doc:`batch_provider`
    Batch provider - for parallel batch data processing.

:doc:`checkpointer`
    Checkpointer - saving/restoring of model/optimizers/schedulers/custom data

:doc:`run`
    Run - helper for launching distributed parallel training

:doc:`download`
    Download - module for downloading and unpacking files.

:doc:`shuffle`
    Shuffle functions for ndarrays.

:doc:`timer`
    Decorator for measuring time

:doc:`epoch`
    Utils for organizing epoch iterations, printing progress, computing average of losses.
	
:doc:`measures`
    Some specific measures not available out of the box in other packages, e.g. F1 measure for open set problems.

:doc:`cache`
    Decorator  for caching return of functions to pickles.

:doc:`async`
    Decorator for concurrent execution.

:doc:`reader`
    Readers from binary MNIST, CIFAR-10, CIFAR-100.

	
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
