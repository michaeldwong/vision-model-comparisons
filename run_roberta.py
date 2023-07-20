
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from torch.nn import functional as F


from transformers import pipeline
from pprint import pprint
import time
from datetime import datetime


num_iters = 100
batch_size = 8

model = 'roberta-large-mnli'
in_text = "This restaurant is awesome"
task_pipe = pipeline("text-classification", model=model, device=0)
output = task_pipe(in_text)



# Text classification
model = 'roberta-large-mnli'
in_text = "This restaurant is awesome"
in_texts = []

for i in range(0, batch_size):
    in_texts.append(in_text)

time.sleep(10)
print('** Roberta text classification **')

print('STARTING AT ', datetime.now())
t0 = time.time()
for i in range(0,num_iters):
#        start = time.time()
    task_pipe = pipeline("text-classification", model=model, device=0)
    output = task_pipe(in_texts)
#        print('Iteration ' , i , dt_start)
#        print('End at ', dt_end)
#        end = time.time()
#        iteration_total = end - start
#        print('elapsed: ', iteration_total)
#        print()

t1 = time.time()
total = t1-t0
print('ENDING AT ', datetime.now())
print('total ', total)
print()

