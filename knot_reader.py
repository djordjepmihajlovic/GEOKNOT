import numpy as np
from knot_init import *
from pathlib import Path
import csv
from knot_invs import *

def read_array(knot):
    state = np.zeros((100, 100, 100), dtype=np.int64)
    for i in knot:
        state[round(i[1])][round(i[2])][round(i[3])] = i[0] # snap to integer values 
    return state

def read_coord(knot):
    coord_list = [(float(i[0]), float( i[1]), float(i[2]), float(i[3])) for i in knot]
    coord_list = sorted(coord_list, key=lambda x: x[0])
    return coord_list

def read_and_concatenate(knot_path, knot_type):
    """
    Read all {knot}_*.csv files from knot_path and concatenate into one list.
    """
    knot_dir = Path(knot_path)
    csv_files = sorted(knot_dir.glob('0_1_*.csv'))
    
    if not csv_files:
        print(f"No files matching '0_1_*.csv' found in {knot_path}")
        return None
    
    all_rows = []
    header = None
    
    for file in csv_files:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            file_header = next(reader)
            if header is None:
                header = file_header
                all_rows.append(header)
            
            # Check constraints for this file
            if not check_constraints(file, knot_type):
                print(f"Skipping {file.name} - failed constraints")
                continue
            
            all_rows.extend(reader)
    
    return all_rows

def check_constraints(file_path, knot_type):
    """
    Check if file satisfies topology and length constraints.
    """
    state = np.loadtxt(file_path, delimiter=',')
    plot_3d_line(state)

    # need to check that each file is length 100
    # need to check that each file is of the required topology.
    topo = Q_invariant(state, 'Uq(sl2)').alexander_polynomial_hash(knot_type, joggle=False) 
    length = len(state)


    if topo and length == 100:
        return True

    else:
        return False
    

check_constraints('/Users/s1910360/Desktop/0_1/0_1_19_16_2321.csv', '0_1')