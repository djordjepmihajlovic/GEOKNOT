import matplotlib.pyplot as plt
from collections import defaultdict
from copy import copy

# make gauss code suitable for attaching handles
def transform_gauss_code(gi):
    n = len(gi)//2 
    # Subsequently reverse order of sequence between i and i for each i
    g=copy(gi)  
    for i in range(1, n + 1):
        first = g.index(i)
        second = g[first+1:].index(i)+first+1
        # reverse the section between them
        g[first + 1:second] = reversed(g[first + 1:second])    
    return g

g_signed=[-1,2,3,-4,5,-6,7,-3,-8,1,-9,-7,4,-5,6,9,-2,8]
#g_signed=[1,-2,3,-1,2,-3]
#g_signed=[1,-2,2,-1]

g=[abs(i) for i in g_signed]
h=transform_gauss_code(g)
print('transformed gauss code',h)

# Position of the letters of the transformed gauss code and initial arcs along the midline
def pos_initial_edges(h):
    n = len(h) // 2
    y_center = 2*n
    Arcs=[]
    x_pos={h[i]:[] for i in range(len(h))}
    for i in range(len(h)):
        x_pos[h[i]].append([(2*i+1,y_center),(2*i+2,y_center)])
    for i in range(0,4*n+1,2):
        Arcs.append(((i,y_center),(i+1,y_center)))
    return x_pos, Arcs

Arcs=pos_initial_edges(h)[1]
x_pos=pos_initial_edges(h)[0]
print('pos of cross',x_pos)
print('Initial segments',Arcs)

# positions of partner indices for each number ID in the code h
partners = defaultdict(list)
for idx, hid in enumerate(h):
    partners[hid].append(idx)

print('partners',partners)

# We will distribute the letters of the gauss code into either the upper half plane or lower half plane
# in terms of attaching handles
def can_be_together(interval1, interval2):
    a1, b1 = interval1
    a2, b2 = interval2
    return not ((a1 < a2 < b1 < b2) or (a2 < a1 < b2 < b1))

def partition_intervals(interval_dict):
    ab_set, be_set = [], []   #above/below 
    keys = list(interval_dict.keys())
    if not keys:
        return ab_set, be_set 
    # Put the first key in "above" set
    first_key = keys[0]
    ab_set.append(first_key)
    # Assign the rest according to relative positions
    for key in keys[1:]:
        interval = interval_dict[key]        
        # Check if it can go into "above" set
        if all(can_be_together(interval_dict[k], interval) for k in ab_set):
            ab_set.append(key)
        else:
            be_set.append(key)    
    return ab_set, be_set


ab_set, be_set  = partition_intervals(partners)
print('Above, Below :',ab_set,be_set)

n = len(h) // 2
y_center = 2*n

# create the handles connecting symbols in a nested fashion
# cross_v keeps track of vertives which are part of edges participating in crossings
cross_v={}
# step keeps a count of the depths covered per handle
step=1
for i in reversed(ab_set):
    arclist=[]
    arc1_i=x_pos[i][0][0]
    arc1_f=x_pos[i][1][1]
    arc2_i=x_pos[i][0][1]
    arc2_f=x_pos[i][1][0]
    arclist+=[(arc2_i,(arc2_i[0],arc2_i[1]+step)),((arc2_i[0],arc2_i[1]+step),(arc1_f[0],arc1_f[1]+step+1)),((arc1_f[0],arc1_f[1]+step+1),arc1_f)]
    arclist+=[(arc1_i,(arc1_i[0],arc1_i[1]+step+1)),((arc1_i[0],arc1_i[1]+step+1),(arc2_f[0],arc2_f[1]+step)),((arc2_f[0],arc2_f[1]+step),arc2_f)]
    Arcs.extend(arclist)
    cross_vl=[(arc2_i[0],arc2_i[1]+step),(arc1_f[0],arc1_f[1]+step+1),(arc1_i[0],arc1_i[1]+step+1),(arc2_f[0],arc2_f[1]+step)]
    for j in cross_vl:
        cross_v[j]=i
    step+=2

step=-1
for i in reversed(be_set):
    arclist=[]
    arc1_i=x_pos[i][0][0]
    arc1_f=x_pos[i][1][1]
    arc2_i=x_pos[i][0][1]
    arc2_f=x_pos[i][1][0]
    arclist+=[(arc2_i,(arc2_i[0],arc2_i[1]+step)),((arc2_i[0],arc2_i[1]+step),(arc1_f[0],arc1_f[1]+step-1)),((arc1_f[0],arc1_f[1]+step-1),arc1_f)]
    arclist+=[(arc1_i,(arc1_i[0],arc1_i[1]+step-1)),((arc1_i[0],arc1_i[1]+step-1),(arc2_f[0],arc2_f[1]+step)),((arc2_f[0],arc2_f[1]+step),arc2_f)]
    Arcs.extend(arclist)
    cross_vl=[(arc2_i[0],arc2_i[1]+step),(arc1_f[0],arc1_f[1]+step-1),(arc1_i[0],arc1_i[1]+step-1),(arc2_f[0],arc2_f[1]+step)]
    for j in cross_vl:
        cross_v[j]=i
    step-=2

print('cross',cross_v)

# to close the diagram, a final handle needs to be added
final_handle=[(pos_initial_edges(h)[1][0][0],(pos_initial_edges(h)[1][0][0][0],pos_initial_edges(h)[1][0][0][1]+step)),((pos_initial_edges(h)[1][0][0][0],pos_initial_edges(h)[1][0][0][1]+step),(pos_initial_edges(h)[1][-1][1][0],pos_initial_edges(h)[1][-1][1][1]+step)), ((pos_initial_edges(h)[1][-1][1][0],pos_initial_edges(h)[1][-1][1][1]+step),pos_initial_edges(h)[1][-1][1])]
Arcs.extend(final_handle)
print(Arcs)

# order the vertices wrt how we walk along the diagram
def order_vertices(edges):
    # Build adjacency list of vertices
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    print('ADJ',adj)
    # starting point 
    start=list(adj)[0]
    print('start',start)
    path = [start]
    prev, curr = None, start
    # Walk through connected vertices
    while True:
        next_nodes = [n for n in adj[curr] if n != prev]
        print(curr)
        print(next_nodes)
        print('')
        if not next_nodes:
            break
        nxt = next_nodes[0]
        path.append(nxt)
        prev, curr = curr, nxt
        if curr == start:
            break
    return path


coords=order_vertices(Arcs)
print('ordered coords',coords)

# Word from a walk and new coords
w=[]
new_coords=[]

g_signed2=[]
for i in g_signed:
    g_signed2.append(i)
    g_signed2.append(i)
print('GS2',g_signed2)
count=0
for p in coords:
    if p in cross_v:
        if len(w)==0 or w[-1]!=cross_v[p]:
            w.append(cross_v[p])
        if g_signed2[count]>0:
            new_coords.append((p[0],p[1],1))
            count+=1
        elif g_signed2[count]<0:
            new_coords.append((p[0],p[1],-1))
            count+=1
    else:
        new_coords.append((p[0],p[1],0))

print('N',new_coords)
print('Word',len(w),w)
print('original gauss code',g)
print('Transformed GC',h)


# Plots a knot/link in 3-space given its coordinates

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

z = [p[2] for p in new_coords]
x = [p[0] for p in new_coords]
y = [p[1] for p in new_coords]

ax.plot3D(x, y, z, 'green')
ax.scatter(x, y, z, 'green')
ax.set_title('3D Line Plot')
plt.show()


######### PLOT SEGMENTS
# #Plot the Knot
# plt.axes(projection='3d')

# # Plot each segment
# for (x1, y1), (x2, y2) in Arcs:
#     plt.plot([x1, x2], [y1, y2], 'b-o')  # 'b-o' means blue line with circle markers

# plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio equal
# plt.grid(True)
# plt.show()
