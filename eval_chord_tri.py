import numpy as np
import matplotlib.pyplot as plt
from chord_compute import *
from trivalent_compute_bn import *
from quantum_knot_invs import *

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


def homotopy(config1, config2, t):
    config = config1 * t + config2 * (1 - t)

    return config

inter_config_1 = homotopy(config1=conf_t1, config2=unknot_t1, t=0.9)
inter_config_2 = homotopy(config1=conf_t2, config2=unknot_t2, t=0.9)
inter_config_3 = homotopy(config1=conf_t3, config2=unknot_t3, t=0.0)

unknot = np.row_stack((inter_config_3, inter_config_2, inter_config_1))

hist_ch = []
hist_cor = []

time = np.linspace(0.9, 1, 100)
for ti in time:
    
    ### Square tests ###
    inter_config_1 = homotopy(config1=conf_1, config2=unknot_1, t=ti)
    knot = np.row_stack((unknot_4, unknot_3, unknot_2, inter_config_1))

    ### Triangle tests ###
    # inter_config_1 = homotopy(config1=conf_t1, config2=unknot_t1, t =ti)
    # inter_config_2 = homotopy(config1=conf_t2, config2=unknot_t2, t =ti)
    # inter_config_3 = homotopy(config1=conf_t3, config2=unknot_t3, t =ti)
    # knot = np.row_stack((inter_config_3, inter_config_2, inter_config_1))

    factor_x = 1/(8*np.pi**2)
    factor_y = 1/(16*np.pi**2)
    _, Y2 = compute_trivalent_feynman_diagram(knot)
    #Y2 = ordered_triple_integral_segmented(knot, settings=cfg)
    # Y2 = polygonal_triple_integral(knot)

    writhe = compute_chord(knot, knot)
    print(f"writhe = {(1/(4*np.pi))*np.sum(writhe)}")
    X = 0

    # for i in range(0, N):
    #     for j in range(0, N):
    #         if i<j:
    #             for k in range(0, N):
    #                 if j<k:
    #                     for l in range(0, N):
    #                         if k<l:
    #                             X += writhe[i][k] * writhe[j][l]

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

plt.scatter(time, hist_cor, marker='x')
plt.hlines(y= -(1/12), xmin=min(time), xmax=max(time))
plt.plot(time, hist_cor)
plt.show()