# How to run the model?
By using MiniZinc and one of the provided instances (*./python/instances/*).

# How to use the python scripts?
_model\_runner.py_ <br>
It executes a given model on a given instance, both paths as parameters, returning a list of couriers' paths, the minimized distance and the execution time.

_data\_generator.py_ <br>
It is used for generating random instances. It is possible to modify all the constants on top. multi_generator generate one instance until reaching complexity order given, increasingly. multi_mixed_generator generates _instances\_per\_order_ instances until reaching complexity order given, increasingly.

_mzn\_stats.py_ <br>
It generates instances using multi_generator, then compute solutions for given model. At last, pictures a double chart of both minimized distance and computational time variation over complexity orders.

_multi\_model\_runner.py_ <br>
It pictures a chart of variation of average computational times varying complexity orders, generating few instances for each of them. The list _MODELS_ is an array of path of different models to use and measure on the same instances, in order to compare them. To add new models it is enough to add their path to _MODELS_ and a name to show in the legend in _model\_to\_name_ dict. 
