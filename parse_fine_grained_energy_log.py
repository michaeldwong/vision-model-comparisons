import csv
from datetime import datetime

def calculate_cumulative_energy(file_path, start_timestamp, end_timestamp):
    total_energy = 0.0
    previous_timestamp = None
    start_dt = datetime.strptime(start_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    end_dt = datetime.strptime(end_timestamp, "%Y-%m-%d %H:%M:%S.%f")

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        for row in reader:
            timestamp_str, energy_str = row[0], row[1]
            timestamp = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S.%f")
            energy = float(energy_str.replace('W', '').strip())

            if start_dt <= timestamp <= end_dt:
                if previous_timestamp is not None:
                    time_diff = (timestamp - previous_timestamp).total_seconds()
                    total_energy += energy * time_diff

                previous_timestamp = timestamp

    return total_energy


# Usage example
energy_file = 'fine_grained_energy_log.txt'
start_timestamp = '2023-07-17 10:41:37.644590'
end_timestamp = '2023-07-17 10:41:41.123456'
cumulative_energy = calculate_cumulative_energy(energy_file, start_timestamp, end_timestamp)
print(f"Cumulative energy consumption: {cumulative_energy} Watt-seconds")

