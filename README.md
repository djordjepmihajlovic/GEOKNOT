# GEOKNOT: a geometrically biased algorithm to sample spaces of knotted objects.

GEOKNOT is a program to sample chosen geometries for topological objects (specifically knots).

* Quick start guide:

 We use writhe and long range entanglement as the 
``` python knot_sampler.py -sub 2 -no 1 -np 1 -k 0_1 ```
- sub (Int): Number of sub-samples per process (divisions of the specified range for features such as writhe and entanglement).
- no (Int): Number of samples at given range (number of samples for each range of features such as writhe and entanglement).
- np (Int): Number of processes. Useful for using multiple cores to sample a large dataset across chosen geometries.
- k (Str): The knot type to instantiate, currently 0_1, ..., 5_2 are implemented, however any starting (lattice) configuration can be loaded.
- geo (List): A list of targetted geometric properties for the entended curve. For speed of convergence/sampling, one should use a broader range of accepted values.

The above code will therefore sample 8 knots. There are 4 different distributions it will target: as each target distribution is divided into 2 (i.e. low writhe, high writhe & low entanglement, high entanglement), for each of the distributions 2 knots are sampled.

To add your own geometric measurements to be used in the sampler:
* Requirements: The function must take as input ordered knot coordinates, indicating the relevant orientation.