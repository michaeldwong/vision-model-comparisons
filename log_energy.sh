echo "" > energy_log.txt

while [ true ]
do
    sleep 10
    date >> energy_log.txt
#    nvidia-smi -i 0 -q -d power >> energy_log.txt
    nvidia-smi >> energy_log.txt
done


