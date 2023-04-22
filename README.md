# How to run the model?
By using MiniZinc and one the provided instances (*./python/instances/*)

# How to use the python scripts?
_model\_runner.py_ 
It execute a given model on a given instance, both paths as parameters, returning a list of paths, the minimized distance and the execution time

_data\_generator.py_
It is used for generating random instances. It is possible to modify all the constants on top. multi_generator generate one instance until reaching complexity order given, increasingly. multi_mixed_generator generates _instances\_per\_order_ instances until reaching complexity order given, increasingly.

_mzn\_stats.py_
It generates instances using multi_generator, then compute solutions for given model. At last, pictures a double chart of both minimized distance and computational time variation over complexity orders.

_multi\_model\_runner.py_
It pictures a chart of variation of average computational times varying complexity orders, generating few instances for each of them. The list _MODELS_ is an array of path of different models to use and measure on the same instances, in order to compare them.
