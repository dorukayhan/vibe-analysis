import csv
import math


def calculate_jerk(csv_file):
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

    # Calculate velocity
    velocities = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        vx = (x_positions[i] - x_positions[i - 1]) / dt
        vy = (y_positions[i] - y_positions[i - 1]) / dt
        vz = (z_positions[i] - z_positions[i - 1]) / dt
        velocities.append((vx, vy, vz))

    # Calculate acceleration
    accelerations = []
    for i in range(1, len(velocities)):
        dt = timestamps[i] - timestamps[i - 1]
        ax = (velocities[i][0] - velocities[i - 1][0]) / dt
        ay = (velocities[i][1] - velocities[i - 1][1]) / dt
        az = (velocities[i][2] - velocities[i - 1][2]) / dt
        accelerations.append((ax, ay, az))

    # Calculate jerk
    jerk = []
    jerk_magnitudes = []
    for i in range(1, len(accelerations)):
        dt = timestamps[i] - timestamps[i - 1]
        jx = (accelerations[i][0] - accelerations[i - 1][0]) / dt
        jy = (accelerations[i][1] - accelerations[i - 1][1]) / dt
        jz = (accelerations[i][2] - accelerations[i - 1][2]) / dt
        jerk.append((jx, jy, jz))
        jerk_magnitudes.append(math.sqrt(jx ** 2 + jy ** 2 + jz ** 2))

    average_jerk_magnitude = sum(jerk_magnitudes) / len(jerk_magnitudes)

    return average_jerk_magnitude


root = 'TrialData/'
jerk_data = []

for i in range(23):
    data = [[], [], [], []]
    filename = root + str(i + 1) + '/player.csv'
    print(filename)
    row = [calculate_jerk(filename)]
    jerk_data.append(row)

with open("result.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(jerk_data)
