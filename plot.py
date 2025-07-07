import matplotlib.pyplot as plt
from collections import defaultdict


data = []
with open('/Users/s1910360/Desktop/ML for Knot Theory/Lattice_Knots/time_data.csv', 'r') as file:
    for line in file:
        # Split the line into individual data points
        line_data = line.strip().split('],[')
        for item in line_data:
            # Clean up brackets and convert to integers
            cleaned_item = item.replace('[', '').replace(']', '')
            data.append(list(map(int, cleaned_item.split(','))))

# Aggregate z values by (x, y) pairs
aggregated_data = defaultdict(list)
for x, y, z in data:
    aggregated_data[(x, y)].append(z)

# Compute average z for each (x, y) pair
x_values = []
y_values = []
z_values = []
for (x, y), z_list in aggregated_data.items():
    x_values.append(x)
    y_values.append(y)
    z_values.append(sum(z_list) / len(z_list))  # Average z value

# Create a histogram
plt.hist2d(x_values, y_values, weights=z_values, bins=(len(set(x_values)), len(set(y_values))), cmap='viridis', )
plt.colorbar(label='Average Iterations required')
plt.xlabel('Writhe')
plt.ylabel('Entanglement')
plt.title('Writhe Vs. Entanglement Sample Time (ntk)')
plt.show()