echo "" > energy_log.txt

while [ true ]
do
    sleep 5
    date >> energy_log.txt
    nvidia-smi -q -d power >> energy_log.txt
done


