import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths to accuracy and load testing results
accuracy_file = 'accuracy_testing/results/accuracy_results.csv'
load_testing_file = 'load_testing/results/load_testing_results.csv'

# Check if the files exist
if not os.path.exists(accuracy_file) or not os.path.exists(load_testing_file):
    print(f"Error: Ensure both accuracy and load testing files exist.")
    exit()

# Load accuracy data
accuracy_df = pd.read_csv(accuracy_file)
# Load load testing data
load_testing_df = pd.read_csv(load_testing_file)

# Reshape the Load Testing DataFrame (melt it to create a 'Load Level' column)
load_testing_melted = pd.melt(
    load_testing_df, 
    id_vars=['Platform'], 
    var_name='Load Level', 
    value_name='CPU Utilization'
)

# Clean up the 'Load Level' column to remove 'Load ' prefix (to make it numeric)
load_testing_melted['Load Level'] = load_testing_melted['Load Level'].str.replace('Load ', '').astype(int)

# Standardize the 'Platform' column in load testing data to match the accuracy data
platform_mapping = {
    'Amazon EC2 CPU (%)': 'EC2',
    'AWS Lambda CPU (%)': 'Lambda',
    'Google Cloud Run CPU (%)': 'Cloud Run',
    'Google Compute Engine CPU (%)': 'Google Compute'
}

load_testing_melted['Platform'] = load_testing_melted['Platform'].replace(platform_mapping)

# Merge the data based on common columns (Platform and Load Level)
combined_df = pd.merge(accuracy_df, load_testing_melted, on=['Platform', 'Load Level'], how='inner')

# Ensure the results directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# Calculate average metrics (mean) for numeric columns (excluding non-numeric columns like 'Image')
average_metrics = combined_df.groupby(['Platform', 'Load Level']).mean(numeric_only=True).reset_index()

# Print the combined average metrics table
print("\nCombined Average Metrics:")
print(average_metrics)

# Save the combined average metrics table to a CSV
combined_df.to_csv('results/combined_average_metrics.csv', index=False)
print("Combined average metrics table saved as 'combined_average_metrics.csv'.")

# Creating the graph similar to the example
platforms = average_metrics['Platform'].unique()
load_levels = sorted(average_metrics['Load Level'].unique())
colors = ['blue', 'orange', 'green']  # Colors for different load levels
bar_width = 0.15  # Reduced width to make the bars thinner
x = np.arange(len(platforms))  # The label locations

plt.figure(figsize=(12, 8))

# Plot bars for each load level
for i, load in enumerate(load_levels):
    load_data = average_metrics[average_metrics['Load Level'] == load]
    plt.bar(x + i * bar_width, load_data['F1 Score'], width=bar_width, color=colors[i % len(colors)])

# Add labels and title
plt.xlabel('Platform', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Accuracy vs Load Level across Platforms', fontsize=16)
plt.xticks(x + bar_width, platforms)
plt.grid(True)

# Save the graph in the results folder
plt.savefig('results/accuracy_vs_load_across_platforms_combined.png')
plt.show()

# Save the overall averages table as a CSV file
overall_averages = average_metrics.groupby('Platform').mean(numeric_only=True)
overall_averages.to_csv('results/overall_averages.csv', index=True)
print("Overall averages table saved as 'overall_averages.csv'.")
