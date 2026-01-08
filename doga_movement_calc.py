import csv
import math
import pandas as pd

type Vec3 = tuple[float, float, float]

def calculate_position_derivatives(csv_file: str) -> tuple[list[Vec3], list[Vec3], list[Vec3]]:
    timestamps = []
    x_positions = []
    y_positions = []
    z_positions = []

    # read csv file
    # HMDPosition does follow the timestamp-x-y-z format
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if present
        for row in reader:
            try:
                timestamp = float(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                # last line is truncated in trial 30 because the sim froze
                # z throws IndexError and we skip the lines below
                timestamps.append(timestamp)
                x_positions.append(x)
                y_positions.append(y)
                z_positions.append(z)
            except IndexError:
                pass

    # Calculate velocity
    velocities = []
    velocities_magnitude = []
    temp = 0
    dt = 0.0
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[temp] < 1:
            continue

        dt = timestamps[i] - timestamps[temp]
        vx = (x_positions[i] - x_positions[temp]) / dt
        vy = (y_positions[i] - y_positions[temp]) / dt
        vz = (z_positions[i] - z_positions[temp]) / dt
        velocities.append((vx, vy, vz))
        velocities_magnitude.append(math.sqrt(vx ** 2 + vy ** 2 + vz ** 2))
        temp = i - 1

    average_velocity_magnitude = sum(velocities_magnitude) / len(velocities_magnitude)

    # Calculate acceleration
    accelerations = []
    accelerations_magnitude = []
    for i in range(1, len(velocities)):
        ax = (velocities[i][0] - velocities[i - 1][0]) / dt
        ay = (velocities[i][1] - velocities[i - 1][1]) / dt
        az = (velocities[i][2] - velocities[i - 1][2]) / dt
        accelerations.append((ax, ay, az))
        accelerations_magnitude.append(math.sqrt(ax ** 2 + ay ** 2 + az ** 2))

    average_acceleration_magnitude = sum(accelerations_magnitude) / len(accelerations_magnitude)

    # Calculate jerk
    jerk = []
    jerk_magnitudes = []
    for i in range(1, len(accelerations)):
        jx = (accelerations[i][0] - accelerations[i - 1][0]) / dt
        jy = (accelerations[i][1] - accelerations[i - 1][1]) / dt
        jz = (accelerations[i][2] - accelerations[i - 1][2]) / dt
        jerk.append((jx, jy, jz))
        jerk_magnitudes.append(math.sqrt(jx ** 2 + jy ** 2 + jz ** 2))

    average_jerk_magnitude = sum(jerk_magnitudes) / len(jerk_magnitudes)

    # return average_velocity_magnitude, average_acceleration_magnitude, average_jerk_magnitude
    return velocities, accelerations, jerk