import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.colors as mcolors  
from knot_init import plot_3d

knot = np.loadtxt('/Users/s1910360/Desktop/0_1_445.csv', delimiter=',', dtype=int)

def read(knot):
    state = np.zeros((100, 100, 100), dtype=np.int64)
    for i in knot:
        state[i[1]][i[2]][i[3]] = i[0]
    return state

state = read(knot)
plot_3d(state)