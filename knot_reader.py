import numpy as np
from knot_init import *
from pathlib import Path
import csv
from knot_invs import Q_invariant

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

    # with open(f'{knot_type}.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header[1:])
    #     for row in all_rows[1:]:
    #         writer.writerow(row[1:])
    
    with open(f'{knot_type}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in all_rows:
            writer.writerow(row[1:]) 

    return all_rows

def check_constraints(file_path, knot_type):
    """
    Check if file satisfies topology and length constraints.
    """
    knot = np.loadtxt(file_path, delimiter=',')

    state = {tuple(coord[1:]): coord[0] for coord in read_coord(knot)}

    # need to check that each file is length 100
    # need to check that each file is of the required topology.
    topo = Q_invariant(state, 'Uq(sl2)').alexander_polynomial_hash(knot_type, joggle=False) 
    length = len(state)
    print(length)

    if topo and length == 100:
        return True

    else:
        return False
    
# check_constraints('/Users/s1910360/Desktop/3_1/3_1_19_16_2321.csv', '3_1')
# read_and_concatenate('/Users/s1910360/Desktop/0_1', '0_1')
# read_and_concatenate('/Users/s1910360/Desktop/3_1', '3_1')