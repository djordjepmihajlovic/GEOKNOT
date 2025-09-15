import numpy as np
import matplotlib.pyplot as plt
from chord_compute import *
# from trivalent_compute_bn import *
from quantum_knot_invs import *
# from trivalent_compute_log import *
from trivalent_MC import *

'''
This code is to ensure the calculations act as expected on some ideal, smooth embeddings of knots.
'''

def load(knot_type):
    file = f"data/{knot_type}.csv"
    data = np.loadtxt(file)
    Nbeads = 152
    n_cols = 3
    data = data.reshape(-1, Nbeads, n_cols)
    return data

def plot(knot):
    x = [item[0] for item in knot]
    y = [item[1] for item in knot]
    z = [item[2] for item in knot]
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot line
    ax.plot3D(x, y, z, color='gray', linewidth=1.5)
    # Scatter with colors based on values
    ax.scatter(x, y, z, s=10)
    plt.show()

data_tk = load("smalltk")

N = 5
t = np.linspace(0, 2*np.pi, N)
x = np.sin(t) + 2 * np.sin(2*t)
y = np.cos(t) - 2 * np.cos(2*t)
z = np.sin(3*t)
trefoil = np.column_stack((x, y, z))

x = np.sin(t)
y = np.cos(t) 
z = 0*t
unknot = np.column_stack((x, y, z))


hist_ch = []
hist_cor = []
lams = np.linspace(0, 1, 100)
aends = np.linspace(0.01, 1, 1000)
ains = np.linspace(1, 1000, 1000)
plot(unknot)
for xi in range(0, 1):
    knot = unknot
    factor_x = 1/(8*np.pi**2)
    factor_y = 1/(16*np.pi**2)
    _, Y2 = compute_trivalent_feynman_diagram(knot, lam=10)
    #Y2 = ordered_triple_integral_segmented(knot, settings=cfg)
    # Y2 = polygonal_triple_integral(knot)

    writhe = compute_chord(knot, knot)
    print(f"writhe = {(1/(4*np.pi))*np.sum(writhe)}")
    X = 0
    Y2 = 0

    for i in range(0, N):
        for j in range(0, N):
            if i<j:
                for k in range(0, N):
                    if j<k:
                        Y2 += writhe[i][j] * writhe[i][k] * writhe[j][k]
                        for l in range(0, N):
                            if k<l:
                                X += writhe[i][k] * writhe[j][l]

    # for an unknot we have AC polynomial = 1
    # so, the invariant w_2 (Bar Natan) of an unknot = 1/24 + 0 = 0.0416

    # for a trefoil we have AC polynomial = z^{2} + 1
    # so, the invariant w_2 (Bar Natan) of a trefoil = 1/24 + 1 = 1.0416
    # for some reason guangadnini say 1/12? -> this is to do w the gauge group of the theory = 0.083

    X = factor_x * X
    Y2 = factor_y * Y2
    w_2 = X + Y2
    hist_ch.append(X)
    hist_cor.append(w_2)
    print(X, Y2)
    print(w_2)

# plt.hist(hist_ch, alpha=0.7, bins=20, label="chord")
# plt.hist(hist_cor, alpha=0.7, bins=20, label="corrected")
# plt.vlines(x=-1/12, ymin=0, ymax=5)
# plt.legend()
# plt.show()
# plt.title(f"{np.mean(hist_cor)}")
# plt.hlines(np.mean(hist_cor), xmin=aends[0], xmax=aends[-1])
# plt.plot(lams, hist_cor)
# plt.show()