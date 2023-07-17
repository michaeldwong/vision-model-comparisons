
bash log_energy.sh &
python3 run_ssd.py > log-wb1.txt &
python3 run_ssd.py >> log-wb1.txt &
python3 run_retinanet.py >> log-wb1.txt &
python3 run_gpt2.py >> log-wb1.txt &
python3 run_gpt2.py >> log-wb1.txt &
python3 run_t5.py >> log-wb1.txt
