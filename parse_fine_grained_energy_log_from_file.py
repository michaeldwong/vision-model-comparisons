import csv
from datetime import datetime
import sys

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


if len(sys.argv) != 3:
    print("Usage: python3 <csv file> <energy log file>")
    sys.exit(1)

# Extract the command line arguments
energy_file = sys.argv[2]
#energy_file = 'fine_grained_energy_log-b1.txt'

model_to_energies = {}
infile = sys.argv[1]
with open(infile, 'r') as f:
    for line in f.readlines():
        items = line.split(',')
        if len(items) == 0:
            continue
        cumulative_energy = calculate_cumulative_energy(energy_file, items[1], items[2])
        if items[0] not in model_to_energies:
            model_to_energies[items[0]] = []
        model_to_energies[items[0]].append(cumulative_energy)
       
for m in model_to_energies:
    print(m , ' -> ', model_to_energies[m]) 

