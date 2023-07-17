echo "" > energy_log.txt

while [ true ]
do
    sleep 10
    date >> energy_log.txt
#    nvidia-smi -i 0 -q -d power >> energy_log.txt
    nvidia-smi >> energy_log.txt
done
#echo "" > energy_log.txt
#echo "" > temperature_log.txt 
#echo "" > clock_log.txt
#while [ true ]
#do
#    date >> energy_log.txt 
#    date >> temperature_log.txt
#    date >> clock_log.txt
#    nvidia-smi >> energy_log.txt 
#    nvidia-smi -q -d temperature >> temperature_log.txt 
#    nvidia-smi -q -d CLOCK >> clock_log.txt
#    sleep 5 
#done

