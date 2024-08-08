import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def parse_csv(file_path):
    df = pd.read_csv(file_path, on_bad_lines='warn',skiprows=2)  # Skip the first row as header is already known
    # Check if the header is as expected
    # if 'Step' not in df.columns or 'Mode' not in df.columns:
    #     raise ValueError("Invalid header format")

    # Process data rows
    voltage = df['Voltage/V'].astype(float)
    capacity = df['Capacity/uAh'].astype(float)
    current = df['Current/uA'].astype(float)

    return voltage, capacity, current

file_paths = glob.glob("/Users/wesleychang/OneDrive - Drexel University/Chang Lab/General/Group/Data/Landt_cyclers/CoinCell_017_FT_*.csv")
color_map = plt.cm.get_cmap('coolwarm')
normalize = Normalize(vmin=0, vmax=len(file_paths))
scalar_map = ScalarMappable(norm=normalize, cmap=color_map)

for i, file_path in enumerate(file_paths):
    voltage, capacity, current = parse_csv(file_path)
    plt.scatter(capacity, voltage, color=scalar_map.to_rgba(i), marker='o', label=f'File {i+1}')

# Set colorbar
scalar_map.set_array([])
cbar = plt.colorbar(scalar_map, label='File Index')

plt.xlabel('Capacity/uAh')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs Capacity')
plt.legend()
plt.show()
