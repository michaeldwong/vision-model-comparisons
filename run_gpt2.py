
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from torch.nn import functional as F


from transformers import pipeline
from pprint import pprint
import time
from datetime import datetime


num_iters = 20
batch_size = 1

model = "gpt2"
in_text = "Hello, I'm a language model"
task_pipe = pipeline(f"text-generation", model=model, device=0)
output = task_pipe(in_text, max_length=30, num_return_sequences=3)

## Text generation
model = "gpt2"
in_texts = []
in_text = "Hello, I'm a language model"
for i in range(0, batch_size):
    in_texts.append(in_text)

time.sleep(10)
print('GPT2 text generation')
print(datetime.now())
t0 = time.time()
for i in range(0, num_iters):
    task_pipe = pipeline(f"text-generation", model=model, device=0)
    output = task_pipe(in_texts, max_length=30, num_return_sequences=3)

t1 = time.time()
total = t1-t0
print('GPT2 total ', total)
print()

