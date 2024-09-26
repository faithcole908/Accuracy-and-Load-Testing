import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Data
platforms = ['Load (Users)', 'Amazon EC2 CPU (%)', 'AWS Lambda CPU (%)', 'Google Cloud Run CPU (%)', 'Google Compute Engine CPU (%)']
load_10 = [10, 25, 30, 28, 22]
load_50 = [50, 45, 50, 48, 42]
load_100 = [100, 70, 75, 72, 68]

# Bar width (make bars thinner)
bar_width = 0.15
# Positions of bars on x-axis
r1 = np.arange(len(platforms))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Create a directory for results if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# ---------------------
# Save the data to a CSV file for further analysis
# ---------------------
cpu_data = {
    'Platform': platforms,
    'Load 10': load_10,
    'Load 50': load_50,
    'Load 100': load_100
}

# Convert the data to a pandas DataFrame
cpu_df = pd.DataFrame(cpu_data)

# Save the DataFrame to a CSV file
csv_file_path = 'results/load_testing_results.csv'
cpu_df.to_csv(csv_file_path, index=False)
print(f"CPU utilization data saved as '{csv_file_path}'.")

# ---------------------
# Plotting the graph
# ---------------------

plt.figure(figsize=(10, 6))

# Creating bars
plt.bar(r1, load_10, color='blue', width=bar_width, edgecolor='grey', label='Load 10')
plt.bar(r2, load_50, color='orange', width=bar_width, edgecolor='grey', label='Load 50')
plt.bar(r3, load_100, color='green', width=bar_width, edgecolor='grey', label='Load 100')

# Adding labels and title
plt.xlabel('Platform', fontweight='bold')
plt.ylabel('CPU Utilization (%)', fontweight='bold')
plt.title('CPU Utilization vs. Load')

# Adding xticks
plt.xticks([r + bar_width for r in range(len(platforms))], platforms)

# Removing the legend box and labels directly on the bars
for i in range(len(load_10)):
    plt.text(r1[i], load_10[i] + 2, str(load_10[i]), ha='center')
    plt.text(r2[i], load_50[i] + 2, str(load_50[i]), ha='center')
    plt.text(r3[i], load_100[i] + 2, str(load_100[i]), ha='center')

# Save and display the graph
plt.tight_layout()
plt.savefig('results/cpu_utilization_vs_load_modified.png')
plt.show()
