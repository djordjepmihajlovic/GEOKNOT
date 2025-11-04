import numpy as np
import matplotlib.pyplot as plt
import math

# params:
# height of smoothing
# level of smoothing 
# angle dependence of smoothing

def _to3(p):
    p = np.asarray(p, dtype=float).reshape(-1)
    return p

def _unit(v):
    n = np.linalg.norm(v)
    return v / n

def hemisphere(p1, p2, p3, n):

    p1s = _to3(p1); p2 = _to3(p2); p3s = _to3(p3)

    p1 = (p1s + p2)/2
    p3 = (p3s + p2)/2
    # rebuild first and last tails
    x = np.linspace(p1s[0], p1[0], int(n/4))
    y = np.linspace(p1s[1], p1[1], int(n/4))
    z = np.linspace(p1s[2], p1[2], int(n/4))
    tail_1 = np.column_stack((x, y, z))
    x = np.linspace(p3[0], p3s[0], int(n/4))
    y = np.linspace(p3[1], p3s[1], int(n/4))
    z = np.linspace(p3[2], p3s[2], int(n/4))
    tail_2 = np.column_stack((x, y, z))

    # Notice: we want to have p1 shifted to the mid point so that it doesnt interfere with other smoothings occuring
    mid_point = p3 + (p1 - p3)/2
    H = -(1/2)
    height = H * np.linalg.norm(p2-mid_point)

    # Diameter vector and radius
    d13 = p3 - p1
    L = np.linalg.norm(d13)
    r = 0.5 * L
    c = 0.5 * (p1 + p3)

    # Plane normal (unit)
    v1 = p2 - p1
    v2 = p3 - p1
    nrm = np.cross(v1, v2)
    nrm = _unit(nrm)

    # Local orthonormal basis in the plane
    ux = _unit(d13)
    uy = np.cross(nrm, ux)  
    uy = _unit(uy)

    sign = np.sign(np.dot(p2 - c, uy))
    if sign == 0:  # p2 lies on the chord line; default to +uy side
        sign = 1.0
    uy *= sign

    # Circle radius for given sagitta (height)
    R = (L*L) / (8.0 * height) + 0.5 * height

    # Circle center offset from chord midpoint along +uy
    O = c + (R - height) * uy

    # Half-angle of the arc
    # Numerically robust clamp
    sin_theta = np.clip((0.5 * L) / R, -1.0, 1.0)
    cos_theta = np.clip((R - height) / R, -1.0, 1.0)
    # Either arctan2 or arccos works; arctan2 is stable:
    theta = np.arctan2(sin_theta, cos_theta)

    # Sample the arc from p1 (t=-theta) to p3 (t=+theta)
    t = np.linspace(-theta, +theta, int(n/2))
    arc = O + R * (-np.cos(t)[:, None] * uy + np.sin(t)[:, None] * ux)
    return [tail_1, arc, tail_2]

def interpolate(points, N):
    P = np.asarray(points, dtype=float)

    chunks = []
    t = np.linspace(0.0, 1.0, N)  
    for i in range(len(P)):
        a, b = P[i], P[(i + 1)%len(P)]
        seg = a + (b - a) * t[:, None]  # (N-1) rows when endpoint=False
        chunks.append(seg)

    return np.vstack(chunks)

def smooth(points, N):
    # want smooth to have a smoothing level (based on homotopy).
    chunk = []
    lp = len(points)
    for i in range(0, len(points)):
        inter_hemi = hemisphere(points[i], points[(i+1)%lp], points[(i+2)%lp], 2*N)
        hemi = inter_hemi[1]
        
        chunk.append(hemi)
    return np.vstack(chunk)

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

def homotopy(config1, config2, t):
    config = config1 * t + config2 * (1 - t)

    return config

# ntk = load('smalltk')

# kink_knot = interpolate(ntk[0], 99)
# smooth_knot = smooth(ntk[0], 99)

# plot(ntk[0])
# plot(kink_knot)
# plot(smooth_knot)

def homotope_smooth(knot, kink_knot, smooth_knot, size, B, C):
    '''
    level of smoothing function determined from analysis.
    '''
    P = np.asarray(knot, dtype=float)

    angles = []

    for i in range(len(P)):

        a, b, c = P[i], P[(i + 1)%len(P)], P[(i + 2)%len(P)]
    
        vec_1 = b-a
        vec_2 = c-b
        angle = np.cross(vec_1, vec_2)
        angles.append(angle)
        # bear in mind that the angle index i is the smoothing level for kink index i+1

    chunk = []
    for idx, A in enumerate(angles):
        if idx<len(angles)-1:
            # if idx == 3:
            #     emp = []
            #     sm = smooth_knot[(idx)*size:(idx+1)%len(kink_knot)*size]
            #     sm2 = smooth_knot[(idx+1)*size:(idx+2)%len(kink_knot)*size]
            #     ki = kink_knot[(idx)*size + math.floor(size/2):(idx+2)%len(kink_knot)*size - math.floor(size/2)]
            #     print(len(sm))
            #     print(len(ki))
            #     emp.append(sm)
            #     emp.append(sm2)
            #     # emp.append(ki)
            #     plot(np.vstack(emp))

            # smoothing = np.sin(C*A)*B #(0.00506)
            smoothing = 1
            # B and C are learned coeffs
            partial = homotopy(smooth_knot[(idx)*size:(idx+1)%len(kink_knot)*size], kink_knot[(idx)*size + math.floor(size/2):(idx+2)%len(kink_knot)*size - math.ceil(size/2)], smoothing)
            chunk.append(partial)

    return np.vstack(chunk)

# ntk = load('smalltk')
# print(len(interpolate(ntk[0], 14)))
# print(len(smooth(ntk[0], 14)))

# selective_smooth = homotope_smooth(ntk[0], interpolate(ntk[0], 14), smooth(ntk[0], 14), 14, 1, 1)
# plot(selective_smooth)
# plot(homotopy(interpolate(ntk[20], 13), smooth(ntk[20], 13), 0.0001))
# plot(selective_smooth)