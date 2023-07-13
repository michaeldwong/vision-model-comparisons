
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from torch.nn import functional as F


from transformers import pipeline
from pprint import pprint
import time
from datetime import datetime


num_iters = 20
batch_size = 1


model = 't5-base'
in_text = "Hello, how are you doing?"
task_pipe = pipeline("translation_en_to_fr", model=model, max_length=1028, device=0)
output = task_pipe(in_text)



## TRanslation


model = 't5-base'
in_texts = []
in_text = "Hello, how are you doing?"

for i in range(0, batch_size):
    in_texts.append(in_text)

time.sleep(10)
print('T5 translation')
print(datetime.now())
t0 = time.time()
for i in range(0,num_iters):
    task_pipe = pipeline("translation_en_to_fr", model=model, max_length=1028, device=0)
    output = task_pipe(in_texts)
t1 = time.time()
total = t1-t0
print('T5 total ', total)
print()


