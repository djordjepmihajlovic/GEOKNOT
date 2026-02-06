import numpy as np
from knot_init import *
from pathlib import Path
import csv
from knot_invs import Q_invariant

def read_array(knot):
    "Read in knot on array as lattice - notice this might change topology(!)"
    state = np.zeros((100, 100, 100), dtype=np.int64)
    for i in knot:
        state[round(i[1])][round(i[2])][round(i[3])] = i[0] # snap to integer values 
    return state

def read_coord(knot):
    "Read coord in coord state (pos, x, y, z) - pos is position in chain for tracking orientation"
    coord_list = [(float(i[0]), float( i[1]), float(i[2]), float(i[3])) for i in knot]
    coord_list = sorted(coord_list, key=lambda x: x[0])
    return coord_list

def read_and_concatenate(knot_path, knot_type):
    """
    Read all {knot}_*.csv files from knot_path and concatenate into one list. Used for building dataset from sampled knots.
    """
    broken = 0
    total = 0
    knot_dir = Path(knot_path)
    csv_files = sorted(knot_dir.glob(f'{knot_type}_*.csv'))
    
    if not csv_files:
        print(f"No files matching '{knot_type}_*.csv' found in {knot_path}")
        return None
    
    all_rows = []
    
    for file in csv_files:
        with open(file, 'r') as f:
            total += 1
            reader = csv.reader(f)
            
            # Check constraints for this file
            if not check_constraints(file, knot_type):
                print(f"Skipping {file.name} - failed constraints")
                broken += 1
                continue
            
            all_rows.extend(reader)
    
    print(f'Number of knots: {total}.')
    print(f'Number of broken knots: {broken}.')
    print(f'Number of valid knots: {total-broken}.')
    
    with open(f'{knot_type}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in all_rows:
            writer.writerow(row[1:]) 

    return all_rows

def check_constraints(knot, knot_type):
    """
    Check if knot from a file satisfies topology and length constraints.
    """
    orientation = np.arange(0, len(knot))
    coords = [(i[0], i[1], i[2]) for i in knot]
    state = {tuple(coord[0:]): orientation[idx] for idx, coord in enumerate(coords)}
    topo = Q_invariant(state, 'Uq(sl2)').alexander_polynomial_hash(knot_type, joggle=False) 
    length = len(state)

    if topo and length == 100:
        print('req. satisfied')
        return True

    else:
        print('req. failed')
        return False
