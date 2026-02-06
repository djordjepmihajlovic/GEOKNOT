import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from knot_init import plot_3d_line
from knot_reader import check_constraints


def run_check(knot_type='0_1', dist = 'iid_1', index = 0, plot = False):
    "Check sample from generated datasets"
    XYZ = np.loadtxt(f'{knot_type}/{knot_type}_{dist}.csv', delimiter=',')
    XYZ = XYZ.reshape(-1, 100, 3)
    XYZ = XYZ[index]
    if plot:
        'visual confirmation'
        plot_3d_line(XYZ)
    check_constraints(XYZ, knot_type=knot_type)

run_check(index=0) # Check 0th knot in dataset (index)