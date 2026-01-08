import csv
import math


def calculate_path(csv_file):
    timestamps = []
    x_positions = []
    y_positions = []
    z_positions = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if present
        for row in reader:
            timestamp = float(row[0])
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])

            timestamps.append(timestamp)
            x_positions.append(x)
            y_positions.append(y)
            z_positions.append(z)

    # Calculate path length
    path_length = 0.0
    for i in range(1, len(x_positions)):
        dx = x_positions[i] - x_positions[i - 1]
        dy = y_positions[i] - y_positions[i - 1]
        dz = z_positions[i] - z_positions[i - 1]
        segment_length = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        path_length += segment_length

    return [path_length]


root = 'TrialData/'
row = []

subjectNumList = [1,22,50,13,11,25,24,20,12,16,9,4,55,56,57,60,61]
for i in range(len(subjectNumList)):
    data = [[], [], [], []]
    filename = root + "Sub" + str(subjectNumList[i]) + "/CameraMovement.csv"
    print(filename)
    appendList = calculate_path(filename)
    appendList.insert(0, str(subjectNumList[i]))
    row.append(appendList)

with open("HMD_PathResults/result.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row)
