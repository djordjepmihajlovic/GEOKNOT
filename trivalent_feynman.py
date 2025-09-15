import os
import numpy as np
from numba import prange, njit, set_num_threads
from generated_derivatives import *
from sympy import symbols, sqrt, Matrix, diff, lambdify, pycode, pi

'''
Automatic differential version,
Issues:
if a and b are anti parallel (|a||b| + a\dot b) will cause probs.
need to deal with pairwise collisions that make |a| or |b| small.
Central difference tangent vectors are not well implemented for polygonal knot.
Adjacency is when we have blow ups.
'''

def generate_derivatives():
    y1,y2,y3,x1,x2,x3,z1,z2,z3 = symbols('y1 y2 y3 x1 x2 x3 z1 z2 z3')
    x = Matrix([x1, x2, x3])
    y = Matrix([y1, y2, y3])
    z = Matrix([z1, z2, z3])

    dIdydz = [[[None for _ in range(3)] for _ in range(3)] for _ in range(3)]

    for sigma in range(3):
        for jy in range(3):
            for jz in range(3):
                dIdydz[sigma][jy][jz] = diff(Ivec(sigma), z[jz], y[jy])

    dIdydz_func = [[[None for _ in range(3)] for _ in range(3)] for _ in range(3)]
    vars = [y1,y2,y3,x1,x2,x3,z1,z2,z3]

    if os.path.exists("generated_derivatives.py") == False:

        # create derivatives
        # actually currently this code is kind of useless

        outfile = "generated_derivatives.py"
        with open(outfile, "w") as f:
            f.write("import numpy as np\n")
            f.write("import math\n")
            f.write("from numba import njit\n\n")

            for sigma in range(3):
                for jy in range(3):
                    for jz in range(3):
                        dIdydz_func[sigma][jy][jz] = lambdify(vars, dIdydz[sigma][jy][jz], 'numpy')

                        code = pycode(dIdydz[sigma][jy][jz])
                        fname = f"dIdydz_{sigma}_{jy}_{jz}"
                        f.write(f"@njit\n")
                        f.write(f"def {fname}(y1,y2,y3,x1,x2,x3,z1,z2,z3):\n")
                        f.write(f"  return {code}\n\n")

def Ivec(sigma):
    y1,y2,y3,x1,x2,x3,z1,z2,z3 = symbols('y1 y2 y3 x1 x2 x3 z1 z2 z3')
    x = Matrix([x1, x2, x3])
    y = Matrix([y1, y2, y3])
    z = Matrix([z1, z2, z3])
    a = y-x
    b = z-x

    na = sqrt(a.dot(a))
    nb = sqrt(b.dot(b))
    nab = sqrt((a-b).dot(a-b))
    dot_ab = a.dot(b)

    S = (na+nb-nab)/((na*nb)+dot_ab)

    return pi*2*S*(((a[sigma])/na) + ((b[sigma])/nb))

# ## dIdydz_func[sigma][lambda][tau] -> single number

def load_curve(knot_type, Nbeads):
    '''
    Load the data from the knots database
    Milnor triple integral computation
    '''

    fname_sts = f"{knot_type}.csv"
    my_knot_dir = "PyKnotData/data/" # cluster loc
    curve = np.loadtxt(os.path.join(my_knot_dir, fname_sts))
    curve = curve.reshape(-1, Nbeads, 3)
    return curve

@njit(parallel=True)
def compute_trivalent_feynman_diagram(ring1):

    matrix = np.zeros((ring1.shape[0],ring1.shape[0],ring1.shape[0]))
    val = 0
    for x in range(ring1.shape[0]):
        for y in range(ring1.shape[0]):
            for z in prange(ring1.shape[0]):
                if z != y and z != x and x != y:
                    # eps = 1e-2
                    # if np.linalg.norm(ring1[y]-ring1[x])<eps or np.linalg.norm(ring1[z]-ring1[y])<eps or np.linalg.norm(ring1[z]-ring1[x])<eps:
                    #     continue
                    
                    s = 0
                    l = 1
                    t = 2
                    matrix[x,y,z] += I_a(ring1, y, z, x, l, t, s)

                    s = 2
                    l = 0
                    t = 1
                    matrix[x,y,z] += I_a(ring1, y, z, x, l, t, s)

                    s = 1 
                    l = 2
                    t = 0
                    matrix[x,y,z] += I_a(ring1, y, z, x, l, t, s)

                    s = 0
                    l = 2
                    t = 1
                    matrix[x,y,z] -= I_a(ring1, y, z, x, l, t, s)

                    s = 2
                    l = 1
                    t = 0
                    matrix[x,y,z] -= I_a(ring1, y, z, x, l, t, s)

                    s = 1 
                    l = 0
                    t = 2
                    matrix[x,y,z] -= I_a(ring1, y, z, x, l, t, s)

                    if x<y<z:
                        val += matrix[x,y,z]
                    
                else:
                    matrix[x,y,z] = 0
    
    return matrix

@njit()
def I_a(ring1, y, z, x, l, t, s):
    '''
    l, t determine derivative wrt z and y, s determines the index of a and b.
    '''

    dx = ring1[(x+1)%ring1.shape[0]] - ring1[(x-1)%ring1.shape[0]] # dx
    dy = ring1[(y+1)%ring1.shape[0]] - ring1[(y-1)%ring1.shape[0]] # dy
    dz = ring1[(z+1)%ring1.shape[0]] - ring1[(z-1)%ring1.shape[0]] # dz

    # actually need

    x1, x2, x3 = ring1[x]
    y1, y2, y3 = ring1[y]
    z1, z2, z3 = ring1[z]

    w = dx[s] * dy[l] * dz[t]

    # s = 0
    if s == 0 and l == 0 and t == 0:
        return dIdydz_0_0_0(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 0 and l == 0 and t == 1:
        return dIdydz_0_0_1(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 0 and l == 0 and t == 2:
        return dIdydz_0_0_2(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 0 and l == 1 and t == 0:
        return dIdydz_0_1_0(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 0 and l == 1 and t == 1:
        return dIdydz_0_1_1(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 0 and l == 1 and t == 2:
        return dIdydz_0_1_2(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 0 and l == 2 and t == 0:
        return dIdydz_0_2_0(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 0 and l == 2 and t == 1:
        return dIdydz_0_2_1(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 0 and l == 2 and t == 2:
        return dIdydz_0_2_2(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    
    # s = 1
    elif s == 1 and l == 0 and t == 0:
        return dIdydz_1_0_0(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 1 and l == 0 and t == 1:
        return dIdydz_1_0_1(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 1 and l == 0 and t == 2:
        return dIdydz_1_0_2(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 1 and l == 1 and t == 0:
        return dIdydz_1_1_0(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 1 and l == 1 and t == 1:
        return dIdydz_1_1_1(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 1 and l == 1 and t == 2:
        return dIdydz_1_1_2(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 1 and l == 2 and t == 0:
        return dIdydz_1_2_0(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 1 and l == 2 and t == 1:
        return dIdydz_1_2_1(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 1 and l == 2 and t == 2:
        return dIdydz_1_2_2(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    
    # s = 2
    if s == 2 and l == 0 and t == 0:
        return dIdydz_2_0_0(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 2 and l == 0 and t == 1:
        return dIdydz_2_0_1(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 2 and l == 0 and t == 2:
        return dIdydz_2_0_2(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 2 and l == 1 and t == 0:
        return dIdydz_2_1_0(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 2 and l == 1 and t == 1:
        return dIdydz_2_1_1(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 2 and l == 1 and t == 2:
        return dIdydz_2_1_2(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 2 and l == 2 and t == 0:
        return dIdydz_2_2_0(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 2 and l == 2 and t == 1:
        return dIdydz_2_2_1(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w
    elif s == 2 and l == 2 and t == 2:
        return dIdydz_2_2_2(y1,y2,y3,x1,x2,x3,z1,z2,z3) * w

def main():
    knots = ["smalltk_2", "smallntk_2"]

    for idx, x in enumerate(knots):
        set_num_threads(20)
        tri = []
        curves = load_curve(x, 152) # this is quite slow
        print("Manifold loaded")
        print("Calculating possible feynman diagrams...")
        for kth, k in enumerate(curves):
            print(kth)
            volume = compute_trivalent_feynman_diagram(k)
            tri.append(volume)

        tri_array = np.array(tri)  
        np.savez_compressed(f'3DSignedTri_small{x}.npz', tri_array=tri_array)

main()