import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Ensure the 'results' directory exists
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Image URLs
images = [
    "https://i.pinimg.com/736x/32/57/0a/32570ae14dc027d871d8abb0eed6dc31.jpg",
    "https://cf.ltkcdn.net/family/images/orig/200821-2121x1414-family.jpg"
]

# Endpoints for different platforms
endpoints = {
    "EC2": "http://3.83.45.12/analyze",
    "Lambda": "https://rx8wp9i43f.execute-api.us-east-1.amazonaws.com/analyze",
    "Cloud Run": "https://app2imag-943367134170.us-central1.run.app/analyze",
    "Google Compute": "http://35.192.222.124/analyze"
}

# Expected labels (manually obtained based on image content and ground truth)
expected_labels = {
    "https://i.pinimg.com/736x/32/57/0a/32570ae14dc027d871d8abb0eed6dc31.jpg": ["tree", "mountain", "sky"],
    "https://cf.ltkcdn.net/family/images/orig/200821-2121x1414-family.jpg": ["person", "family", "indoor"]
}

# Function to send request to each platform
def send_request(endpoint, image_url, platform):
    try:
        response = requests.post(endpoint, data={"imageUrl": image_url})
        response.raise_for_status()
        return platform, image_url, json.loads(response.text)
    except Exception as e:
        logger.error(f"Error on {endpoint} with image {image_url}: {e}")
        return platform, image_url, []

# Calculate precision, recall, and F1 score
def calculate_metrics(expected, actual):
    expected_set = set(expected)
    actual_set = set([label['Name'].lower() for label in actual])

    true_positives = len(expected_set.intersection(actual_set))
    false_positives = len(actual_set - expected_set)
    false_negatives = len(expected_set - actual_set)

    precision = precision_score([1] * true_positives + [0] * false_positives, [1] * true_positives + [1] * false_positives, zero_division=1)
    recall = recall_score([1] * true_positives + [0] * false_negatives, [1] * true_positives + [0] * false_negatives, zero_division=1)
    f1 = f1_score([1] * true_positives + [0] * false_positives + [0] * false_negatives, [1] * true_positives + [1] * false_positives + [0] * false_negatives, zero_division=1)

    return precision, recall, f1

# Function to test accuracy across platforms
def test_accuracy(load_level=1):
    results = []
    with ThreadPoolExecutor(max_workers=load_level) as executor:
        futures = []
        for image_url in images:
            for platform, endpoint in endpoints.items():
                futures.append(executor.submit(send_request, endpoint, image_url, platform))

        for future in as_completed(futures):
            platform, image_url, result = future.result()

            expected = expected_labels[image_url]
            precision, recall, f1 = calculate_metrics(expected, result)

            results.append({
                "Platform": platform,
                "Image": image_url,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })
    return results

# Prepare dataframes for each load level and save results
def evaluate_load_performance():
    load_levels = [10, 50, 100]
    all_results = pd.DataFrame()

    for load in load_levels:
        logger.info(f"Evaluating performance under load level: {load}")
        result = test_accuracy(load)
        df = pd.DataFrame(result)
        df['Load Level'] = load
        all_results = pd.concat([all_results, df])

    # Save results to CSV in the results folder
    results_path = os.path.join(results_dir, 'accuracy_results.csv')
    all_results.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

    return all_results

# Plotting function for metrics
def plot_metrics(results_df):
    # Group by 'Platform' and 'Load Level', then calculate the mean for only the numeric columns (Precision, Recall, F1 Score)
    grouped_results = results_df.groupby(['Platform', 'Load Level']).mean(numeric_only=True).reset_index()

    platforms = grouped_results['Platform'].unique()
    load_levels = grouped_results['Load Level'].unique()

    # Colors for different platforms
    colors = ['blue', 'orange', 'green', 'red']

    # Plot Precision, Recall, F1 Score for each platform
    for metric in ['Precision', 'Recall', 'F1 Score']:
        plt.figure(figsize=(10, 6))

        bar_width = 0.15  # Thinner bars
        index = np.arange(len(load_levels))  # This is for the load levels (10, 50, 100)

        for i, platform in enumerate(platforms):
            # Filter data for the current platform
            platform_data = grouped_results[grouped_results['Platform'] == platform]
            # Create the bar for each load level
            plt.bar(index + i * bar_width, platform_data[metric], bar_width, color=colors[i])

        plt.title(f'{metric} vs Load Level across Platforms')
        plt.xlabel('Load Level')
        plt.ylabel(metric)
        plt.xticks(index + bar_width * (len(platforms) - 1) / 2, load_levels)  # Position the x-axis labels in the center
        plt.grid(True)

        # Save each plot in the results folder
        plot_path = os.path.join(results_dir, f'{metric.lower().replace(" ", "_")}_vs_load.png')
        plt.savefig(plot_path)
        logger.info(f"{metric} graph saved to {plot_path}")
        plt.close() 



if __name__ == "__main__":
    results_df = evaluate_load_performance()
    plot_metrics(results_df)
