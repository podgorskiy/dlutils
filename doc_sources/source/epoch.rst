Utils for organizing epoch iterations, printing progress, computing average of losses.
======================================================================================

.. autoclass:: dlutils.epoch.EpochRange
   :members:
   :undoc-members:
   :special-members: __len__,  __next__
   
.. autoclass:: dlutils.epoch.LossTracker
   :members:
   :undoc-members:
   :special-members: __str__

.. autoclass:: dlutils.epoch.RunningMean
   :members:
   :undoc-members:
   :special-members: __iadd__,  __float__
