import csv
import os

root_dir = 'TrialData/'
events_file_name = 'events.csv'


# Function to calculate the time difference in minutes
def calculate_time_difference(csv_file_path):
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        timestamps = [float(row[0]) for row in reader]  # Assuming timestamps are in the first column
    return (timestamps[-1] - timestamps[0]) / 60


# Traverse all subdirectories under TrialData
time_differences = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == events_file_name:
            csv_path = os.path.join(dirpath, filename)
            time_diff = calculate_time_difference(csv_path)
            time_differences.append(time_diff)

# Calculate the mean of the time differences
mean_time_difference = sum(time_differences) / len(time_differences)

print("Mean time difference (in minutes):", mean_time_difference)
