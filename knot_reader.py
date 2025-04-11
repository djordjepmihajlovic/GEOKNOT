import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.colors as mcolors  

knot = np.loadtxt('examples/config_3_1_14.csv', delimiter=',', dtype=int)
state = np.empty((100, 100, 100))

for i in knot:
    state[i[1]][i[2]][i[3]] = i[0]


norm = mcolors.Normalize(vmin=np.min(state[state > 0]), vmax=np.max(state))
cmap = cm.coolwarm  

# Initialize color array
colors = np.zeros(state.shape + (4,))  # RGBA color array

# Apply colormap for nonzero values
mask = state > 0  
colors[mask] = cmap(norm(state[mask]))  

colors[..., 3] = np.where(state > 0, 1.0, 0.0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the voxels
ax.voxels(state > 0, facecolors=colors)

ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_zlim([0, 100])

plt.show()