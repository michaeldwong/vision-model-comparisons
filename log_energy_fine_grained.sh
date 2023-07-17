
nvidia-smi --query-gpu=timestamp,power.draw --format=csv --loop-ms=100 --id 0 > fine_grained_energy_log.txt
