# GEOKNOT: a geometrically biased algorithm to sample spaces of knotted objects.

GEOKNOT is a program to sample chosen geometries for topological objects (specifically knots).

### Quick start guide

To run the package you must first build the required conda enviornment, to do so (assuming you already have conda installed), simply run
``` conda env create --file=requirements.yml ```
Once the enviornment is built run 
``` conda activate geoknot ```

We use writhe and long range entanglement as the features being sampled
``` python knot_sampler.py -sub 2 -no 1 -np 1 -k 0_1 -distr [(0, 3), (500, 1000)] -fns [False, False] -plot False -smooth False```
- sub (Int): Number of sub-samples per process (divisions of the specified range for features such as writhe and entanglement).
- no (Int): Number of samples at given range (number of samples for each range of features such as writhe and entanglement).
- np (Int): Number of processes. Useful for using multiple cores to sample a large dataset across chosen geometries.
- k (Str): The knot type to instantiate, currently 0_1 and 3_1 are implemented, however any starting (lattice) configuration can be loaded (see details below).
- fns (List): A list indicating use of functions for sampling. index 0 represents fn_1, and index 1 fn_2. If both set to False (default) then the default geometric features are sampled (fn_1=writhe and fn_2=long-range entanglement). Details for writing own function to sample region below.
- distr (List): A list of targetted geometric property values for the curve. For speed of convergence/sampling, one should use a broader range of accepted values.
- plot (Bool): Plot the desired embedding.
- smooth (Bool): Sample smooth knots (completely off lattice). If set to false knots are sampled in \mathbb{R}^3 by applying a perturbation to the original lattice (required for Alexander determinant computation/consistency with previous ML models trained on polygonal knot embeddings).

* Note:
``` python knot_sampler.py -sub 1 -no 1 -np 1 -k 0_1 -distr [(0, 3), (500, 1000)] -plot True```
will therefore sample a single knot with values in the desired distribution range, it can be useful to use the plot flag here.

The above code will therefore sample 8 knots. There are 4 different distributions it will target: as each target distribution is divided into 2 (i.e. low writhe, high writhe & low entanglement, high entanglement), for each of the distributions 2 knots are sampled.

### Geometric features

To add your own geometric measurements to be used in the sampler:
* Requirements: The function must take as input ordered knot coordinates.
* An example feature is included in knot_functions.py. To use this feature, use the arg -fns[radius_of_gyration, False] (this will bias radius of gyration and long-range entanglement... Don't forget to change the sample distribution accordingly).

### Knot size

Knots are initialized in the knot_init.py file, specifically a unknot and trefoil of size 100 are provided in knot_init.py
Notice: larger knot sizes are more likely to reach required geometries, thus speeding up sample time.