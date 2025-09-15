import numpy as np
import matplotlib.pyplot as plt

# Current goal - fix all the analysis here:

# 1. we need one-loop versus two-loop to be a straight line.
# 2. we need X + Y to do better than just X.

diagrams = [1, 21, 22, "t", 31, 32, 33, 34, 35, "t11", "t12"]
ntk = [[] for i in range(0, len(diagrams))]
tk = [[] for i in range(0, len(diagrams))]
for idx, i in enumerate(diagrams):
    ntk[idx] = np.hstack((np.loadtxt(f'../data/feynman_diagrams/feynman_diagram_smallntk_{i}.csv'), np.loadtxt(f'data/feynman_diagrams/feynman_diagram_smallntk_2_{i}.csv')))
    tk[idx] = np.hstack((np.loadtxt(f'../data/feynman_diagrams/feynman_diagram_smalltk_{i}.csv'), np.loadtxt(f'data/feynman_diagrams/feynman_diagram_smalltk_2_{i}.csv')))
    if type(i) == int:
        if i == 1:
            ntk[idx] /= (152)
            tk[idx] /= (152)
        elif 30>i>20:
            ntk[idx] /= (152*152)
            tk[idx] /= (152*152)
        elif i > 30:
            ntk[idx] /= (152*152*152)
            tk[idx] /= (152*152*152)

    elif i == "t":
        ntk[idx] /= (1)
        tk[idx] /= (1)
    else:
        ntk[idx] /= (1)
        tk[idx] /= (1)

    factor = 1/(32*np.pi**3)
    if type(i) != int:
        ntk[idx] *= factor
        tk[idx] *= factor


### one-loop^{2} == two-loop (X + ||) ###
plt.scatter(np.array(tk[0])**2, np.array(tk[1])+np.array(tk[2]))
plt.xlabel('one-loop contribution')
plt.ylabel('two-loop contribution')
plt.show()

invariant_tk = (np.array(tk[2]))# - (2 * np.array(ntk[3]))
invariant_tk_alt = np.array(tk[2]) - (np.array(tk[3]))
# invariant_tk = ((1/4) * np.array(tk[2])) - (2 * np.array(tk[3]))

w1_tk1 = np.array(ntk[0]) # ((2**2)-1) 
w2_tk1 = np.array(ntk[1]) * 9/2
w2_tk2 = np.array(ntk[2]) * -3/2
w2_tk3 = np.array(ntk[3]) * 6

# invariant_tk_alt = (1/6) * ((w2_tk1+w2_tk2+w2_tk3) - (1/4)*(w2_tk1+w2_tk2))

index = 2
data_min = min(np.min(invariant_tk_alt), np.min(invariant_tk))
data_max = max(np.max(invariant_tk_alt), np.max(invariant_tk))
bins = np.linspace(data_min, data_max, 100)

# plt.hist(ntk[index], alpha=0.5, bins = bins, label="ntk_0")
# plt.hist(tk[index], alpha=0.5, bins = bins, label="tk_0")
plt.hist(invariant_tk, alpha=0.5, bins = bins, label="inv_ntk")
plt.hist(invariant_tk_alt, alpha=0.5, bins = bins, label="inv_ntk_alt")
plt.legend()
plt.show()