import csv

def calculate_cumulative_energy(file_path):
    total_energy = 0.0
    previous_timestamp = None

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        for row in reader:
            timestamp_str, energy_str = row[0], row[1]
            timestamp = float(timestamp_str)
            energy = float(energy_str)

            if previous_timestamp is not None:
                time_diff = timestamp - previous_timestamp
                total_energy += energy * time_diff

            previous_timestamp = timestamp

    return total_energy

# Usage example
energy_file = 'fine_grained_energy_log.txt'
cumulative_energy = calculate_cumulative_energy(energy_file)
print(f"Cumulative energy consumption: {cumulative_energy} Watt-seconds"
