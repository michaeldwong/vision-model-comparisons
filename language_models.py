from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from torch.nn import functional as F


from transformers import pipeline
from pprint import pprint
import time
from datetime import datetime


num_iters = 10
batch_size = 8

model = "gpt2"
in_text = "Hello, I'm a language model"
task_pipe = pipeline(f"text-generation", model=model, device=0)
output = task_pipe(in_text, max_length=30, num_return_sequences=3)

model = 't5-base'
in_text = "Hello, how are you doing?"
task_pipe = pipeline("translation_en_to_fr", model=model, max_length=1028, device=0)
output = task_pipe(in_text)

model = 'roberta-large-mnli'
in_text = "This restaurant is awesome"
task_pipe = pipeline("text-classification", model=model, device=0)
output = task_pipe(in_text)

model = "stevhliu/my_awesome_billsum_model"
in_text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
task_pipe = pipeline("summarization", model=model, device=0)
output = task_pipe(in_text)

with open('per_iteration_timestamps.csv', 'a') as f_per_iteration:
    ## Text generation
    model = "gpt2"
    in_texts = []
    in_text = "Hello, I'm a language model"
    for i in range(0, batch_size):
        in_texts.append(in_text)

    time.sleep(10)
    print('** GPT2 text generation **')
    print('STARTING AT ', datetime.now())
    t0 = time.time()
    for i in range(0, num_iters):
#        start = time.time()
        dt_start = datetime.now()
        task_pipe = pipeline(f"text-generation", model=model, device=0)
        output = task_pipe(in_texts, max_length=30, num_return_sequences=3)
        dt_end = datetime.now()
#        print('Iteration ' , i , dt_start)
#        print('End at ', dt_end)
#        end = time.time()
#        iteration_total = end - start
#        print('elapsed: ', iteration_total)
#        print()
        f_per_iteration.write(f'{model},{dt_start},{dt_end}\n')

    t1 = time.time()
    total = t1-t0
    print('ENDING AT ', datetime.now())
    print('total ', total)
    print()

    ## TRanslation


    model = 't5-base'
    in_texts = []
    in_text = "Hello, how are you doing?"

    for i in range(0, batch_size):
        in_texts.append(in_text)

    time.sleep(10)
    print('** T5 translation **')
    print('STARTING AT ', datetime.now())
    t0 = time.time()
    for i in range(0,num_iters):
#        start = time.time()
        dt_start = datetime.now()
        task_pipe = pipeline("translation_en_to_fr", model=model, max_length=1028, device=0)
        output = task_pipe(in_texts)
        dt_end = datetime.now()
#        print('Iteration ' , i , dt_start)
#        print('End at ', dt_end)
#        end = time.time()
#        iteration_total = end - start
#        print('elapsed: ', iteration_total)
#        print()
        f_per_iteration.write(f'{model},{dt_start},{dt_end}\n')
    t1 = time.time()
    total = t1-t0
    print('ENDING AT ', datetime.now())
    print('total ', total)
    print()


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
        dt_start = datetime.now()
        task_pipe = pipeline("text-classification", model=model, device=0)
        output = task_pipe(in_texts)
        dt_end = datetime.now()
#        print('Iteration ' , i , dt_start)
#        print('End at ', dt_end)
#        end = time.time()
#        iteration_total = end - start
#        print('elapsed: ', iteration_total)
#        print()
        f_per_iteration.write(f'{model},{dt_start},{dt_end}\n')

    t1 = time.time()
    total = t1-t0
    print('ENDING AT ', datetime.now())
    print('total ', total)
    print()


    # Text summarization
    model = "stevhliu/my_awesome_billsum_model"

    in_text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
    in_texts = []

    for i in range(0, batch_size):
        in_texts.append(in_text)

    time.sleep(10)
    print('** Text summarization **')

    print('STARTING AT ', datetime.now())
    t0 = time.time()
    for i in range(0,num_iters):
#        start = time.time()
        dt_start = datetime.now()
        task_pipe = pipeline("summarization", model=model, device=0)
        output = task_pipe(in_texts)
#        print('Iteration ' , i , dt_start)
#        print('End at ', dt_end)
#        end = time.time()
#        iteration_total = end - start
#        print('elapsed: ', iteration_total)
#        print()
        f_per_iteration.write(f'{model},{dt_start},{dt_end}\n')
    t1 = time.time()
    total = t1-t0
    print('ENDING AT ', datetime.now())
    print('total ', total)
    print()

# Question answering
#
#context = "Extractive Question Answering is the task of extracting an answer from a text given a question."
#query = "What is extractive question answering?"
#model = "distilbert-base-cased-distilled-squad"
#for i in range(0, 1):
#    task_pipe = pipeline("question-answering", model=model,  device=0)
#    result = task_pipe(question=query, context=context, model=model, revision="1.0", device=0)
#print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")




## Next sentence prediction
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
#prompt = "The child came home from school."
#next_sentence = "He played soccer after school."
#
#device = 'cuda:0'
#
## Encode the input text
#encoding = tokenizer.encode_plus(prompt, next_sentence, return_tensors='pt')
#
## Move the input tensors to the GPU
#input_ids = encoding['input_ids'].to(device)
#token_type_ids = encoding['token_type_ids'].to(device)
#attention_mask = encoding['attention_mask'].to(device)
#
## Perform the forward pass on the GPU
#outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
#
#outputs = model(**encoding)[0]
#softmax = F.softmax(outputs, dim=1)
#print(softmax)
