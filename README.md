# GEOKNOT: a geometrically biased algorithm to sample spaces of knotted objects.

GEOKNOT is a program to sample chosen geometries for topological objects (specifically knots).

* Quick start guide:
``` python knot_sampler.py -no 10 -sub 20 -np 10 -k 0_1 ```
- no (Int): Number of ###
- sub (Int): Number of sub-divisions of the chosen geometric property & range; if 1 single knot will be sampled between specified range, if 2 range is divided into 2 halves from which two knots are sampled.
- np (Int): Number of processes. Useful for using multiple cores to sample a large dataset across chosen geometries.
- k (Str): The knot type to instantiate, currently 0_1, ..., 5_2 are implemented, however any starting (lattice) configuration can be loaded.
- geo (List): A list of targetted geometric properties for the entended curve. For speed of convergence/sampling, one should use a broader range of accepted values.

To add your own geometric measurements:
* Requirements: The function must take as input ordered knot coordinates, indicating the relevant orientation

TO DO:
Finish read me, 
make code portable/clean up nicely.