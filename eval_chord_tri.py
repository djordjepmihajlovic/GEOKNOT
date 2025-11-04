import numpy as np
import matplotlib.pyplot as plt
from chord_compute import *
from trivalent_compute_bn import *
from quantum_knot_invs import *
from homotopy import *

'''
This code is to ensure the calculations act as expected on some ideal, smooth embeddings of knots.
Want to write a function that takes in a circle and adds a kink of chosen degree which we will homotope towards
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
    min_range = min(min(x), min(y), min(z))
    max_range = max(max(x), max(y), max(z))
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot line
    ax.plot3D(x, y, z, color='gray', linewidth=1.5)
    # Scatter with colors based on values
    ax.scatter(x, y, z, s=10)
    ax.set_xlim([min_range, max_range])
    ax.set_ylim([min_range, max_range])
    ax.set_zlim([min_range, max_range])
    plt.show()

data_tk = load("smalltk")

N = 100
t = np.linspace(0, 1.5*np.pi, N)
x = np.sin(t) + 2 * np.sin(2*t)
y = np.cos(t) - 2 * np.cos(2*t)
z = np.sin(3*t)
trefoil = np.column_stack((x, y, z))

x = np.sin(t)
y = np.cos(t) 
z = 0*t

unknot = np.column_stack((x, y, z))

############## Fourth kink ###############
tx = np.linspace(-1, 0, 25)
ty = np.linspace(0, 1, 25)
x = tx
y = 1+(0*tx)
z = 0*tx
extra = np.column_stack((x, y, z))

x = -1+(0*ty)
y = ty
z = 0*tx
extra2 = np.column_stack((x, y, z))
conf_1 = np.row_stack((extra2, extra))

############## Third kink ###############
tx = np.linspace(0, -1, 25)
ty = np.linspace(-1, 0, 25)
x = -1+(0*ty)
y = ty
z = 0*tx
extra = np.column_stack((x, y, z))

x = tx
y = -1+(0*tx)
z = 0*tx
extra2 = np.column_stack((x, y, z))
conf_2 = np.row_stack((extra2, extra))

############## Second kink ###############
tx = np.linspace(1, 0, 25)
ty = np.linspace(0, -1, 25)
x = tx
y = -1+(0*tx)
z = 0*tx
extra = np.column_stack((x, y, z))

x = 1+(0*ty)
y = ty
z = 0*tx
extra2 = np.column_stack((x, y, z))
conf_3 = np.row_stack((extra2, extra))

############## First kink ###############
tx = np.linspace(0, 1, 25)
ty = np.linspace(1, 0, 25)
x = tx
y = 1+(0*tx)
z = 0*tx
extra = np.column_stack((x, y, z))

x = 1+(0*ty)
y = ty
z = 0*tx
extra2 = np.column_stack((x, y, z))
conf_4 = np.row_stack((extra, extra2))

#########################################

############## Square wedges ############

# wedge 1
tf = np.linspace(1.5*np.pi, 2*np.pi, 50)
x = np.sin(tf)
y = np.cos(tf)
z = 0*tf
unknot_1 = np.column_stack((x, y, z))
# wedge 2
tf = np.linspace(1*np.pi, 1.5*np.pi, 50)
x = np.sin(tf)
y = np.cos(tf)
z = 0*tf
unknot_2 = np.column_stack((x, y, z))
# wedge 3
tf = np.linspace(0.5*np.pi, 1*np.pi, 50)
x = np.sin(tf)
y = np.cos(tf)
z = 0*tf
unknot_3 = np.column_stack((x, y, z))
# wedge 4
tf = np.linspace(0*np.pi, 0.5*np.pi, 50)
x = np.sin(tf)
y = np.cos(tf)
z = 0*tf
unknot_4 = np.column_stack((x, y, z))

############## Triangle wedges ############

# wedge 1
tf = np.linspace(1.5*np.pi, 2*np.pi, 25)
x = np.sin(tf)
y = np.cos(tf)
z = 0*tf
unknot_t1 = np.column_stack((x, y, z))
# wedge 2
tf = np.linspace(1*np.pi,1.5*np.pi, 25)
x = np.sin(tf)
y = np.cos(tf)
z = 0*tf
unknot_t2 = np.column_stack((x, y, z))
# wedge 3
tf = np.linspace(0*np.pi, 1*np.pi, 50)
x = np.sin(tf)
y = np.cos(tf)
z = 0*tf
unknot_t3 = np.column_stack((x, y, z))

############## Third kink ###############
ty = np.linspace(1, -1, 50)
x = 0*ty
y = ty
z = 0*ty
conf_t3 = np.column_stack((x, y, z))

############## First kink ###############
tx = np.linspace(0, -1, 25)
ty = np.linspace(-1, 0, 25)
x = tx
y = ty
z = 0*tx
conf_t2 = np.column_stack((x, y, z))

############## Second kink ###############
tx = np.linspace(-1, 0, 25)
ty = np.linspace(0, 1, 25)
x = tx
y = ty
z = 0*tx
conf_t1 = np.column_stack((x, y, z))

def arc(angle, radius, n=100, return_smoothed=True):
    """
    Builds a filleted corner between +x and a ray at `angle` (radians).
    Ensures: leg1 has ~n/4 pts, arc has ~n/2 pts, leg2 has ~n/4 pts.
    Total returned points = n.
    """
    if not (0 < angle < np.pi):
        raise ValueError("angle must be in (0, pi).")
    n = int(n)
    if n < 4:
        raise ValueError("n must be >= 4 to allocate points to both legs and the arc.")

    # ---- allocate counts: 1/4 + 1/2 + 1/4 ----
    n_leg = max(1, (n // 2)-2)         # per leg
    n_arc  = max(2, n - 2 * n_leg)  # remaining to the arc, at least 2
    # If there's any remainder (due to rounding), add it to the arc
    n_arc += n - (2 * n_leg + n_arc)  # keeps total exactly n

    # ---- geometry ----
    u1 = np.array([1.0, 0.0])
    u2 = np.array([np.cos(angle), np.sin(angle)])
    b = u1 + u2
    b /= np.linalg.norm(b)

    C = b * (radius / np.sin(angle / 2.0))      # arc center
    t = radius / np.tan(angle / 2.0)            # distance to tangency
    T1 = u1 * t                                 # tangency on +x
    T2 = u2 * t                                 # tangency on rotated ray

    a1 = np.arctan2(T1[1] - C[1], T1[0] - C[0])
    a2 = np.arctan2(T2[1] - C[1], T2[0] - C[0])
    if a2 < a1:
        a2 += 2*np.pi

    # ---- sample (exclude T1/T2 on legs, include them on the arc) ----
    # leg1: from origin -> T1, exclude T1 to avoid duplication
    if n_leg > 0:
        t1 = np.linspace(0.0, 1.0, n_leg, endpoint=False)  # includes 0, excludes 1
        leg1 = np.column_stack((t1 * T1[0], t1 * T1[1], np.zeros_like(t1)))
    else:
        leg1 = np.zeros((0, 3))

    # arc: from T1 to T2, include both ends
    th = np.linspace(a1, a2, n_arc, endpoint=True)
    arc_pts = np.column_stack((C[0] + radius*np.cos(th),
                               C[1] + radius*np.sin(th),
                               np.zeros_like(th)))

    # leg2: from T2 -> origin, exclude T2 to avoid duplication, include origin
    if n_leg > 0:
        s2 = np.linspace(0.0, 1.0, n_leg + 1)[1:]  # (0,1] so T2 excluded, origin included
        leg2_xy = (1.0 - s2)[:, None] * T2  # linear blend from T2 to 0
        leg2 = np.column_stack((leg2_xy, np.zeros_like(s2)))
    else:
        leg2 = np.zeros((0, 3))

    poly = np.vstack((leg1, arc_pts, leg2))  # length == n

    if not return_smoothed:
        return poly

    # optional full circle with the same total number of points
    th_full = np.linspace(a1, a1 + 2*np.pi, n, endpoint=False)
    circle = np.column_stack((C[0] + radius*np.cos(th_full),
                              C[1] + radius*np.sin(th_full),
                              np.zeros_like(th_full)))
    return poly, circle


# angle  = np.deg2rad(20)  
# radius = 0.25
# kinked, circle = arc(angle, radius, n=50)

# unknot = homotopy(config1=circle, config2=kinked, t=0.1)

# plot(unknot)

time = np.linspace(0, 0.1, 10000)

angles = np.arange(100, 5, -5)
print(angles)
plots = [[], []]
sample_data = [[], [], [], []]

ns = [13, 11]

knots = load('smallntk')

Bs = [1]
Cs = [1]
plots = []

for C in Cs:
    for B in Bs:
        data = []

        for xi in range(0,10):

            tknot = homotope_smooth(knots[xi], interpolate(knots[xi], 3), smooth(knots[xi], 3), 3, B, C)
            # must be even numbers for smoothing reasons// shared vertices

            factor_x = 1/(8*np.pi**2)
            factor_y = 1/(16*np.pi**2)
            Y2 = compute_trivalent_feynman_diagram(tknot)
            Y2alt = compute_trivalent_feynman_diagram(knots[xi])

            writhe = compute_chord(knots[xi], knots[xi])
            plt.imshow(writhe)
            plt.show()
            print(f"writhe = {(1/(4*np.pi))*np.sum(writhe)}")
            X = 0

            for i in range(0, N):
                for j in range(0, N):
                    if i<j:
                        for k in range(0, N):
                            if j<k:
                                for l in range(0, N):
                                    if k<l:
                                        X += writhe[i][k] * writhe[j][l]

            X = factor_x * X
            Y2 = factor_y * Y2
            Y2alt = factor_y * Y2

            w_2 = X + Y2
            print(f"smoothed: {w_2}")
            w_2alt = X + Y2alt
            print(f"original: {w_2alt}")

            true_val = -(1/2)

            convergence = abs(Y2-true_val)
            convergencealt = abs(Y2alt-true_val)

            # print(convergencealt-convergence)
            data.append(convergencealt-convergence)

        plots.append(data)

for x in range(0, len(plots)):
    plt.hist(plots[x])

# for i in range(0, len(angles)):
#     plt.scatter(time, sample_data[i], marker='x', label=angles[i])
#     plt.plot(time, sample_data[i])
# plt.hlines(y= -(1/12), xmin=min(time), xmax=max(time))
# plt.legend()
# plt.show()

# for x in range(0, len(plots)):
#     plt.plot(angles, plots[x], label=f"Samples: {ns[x]}")
# plt.ylabel('Smoothing')
# plt.xlabel('Kink angle')
# plt.legend()
# plt.show()