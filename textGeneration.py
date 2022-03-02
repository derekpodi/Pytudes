#!/usr/bin/env python3

#Program to produce generated text via Transformers
#Run in Virtual Enviroment

#https://github.com/nicknochnack/GPTNeo/blob/main/GPT%20Neo%20Tutorial.ipynb


### Setup Venv ENV
# $python3 -m venv env
# $source env/bin/activate
# $(env) pip install --upgrade pip
# $(env) pip3 install torch torchvision torchaudio
# $(env) pip3 install transformers

### Transformers GPT-Neo Text Generation

from transformers import pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B') #Long Load

prompt = "The current stock market" #Variable Text Prompt

res = generator(prompt, max_length=50, do_sample=True, temperature=0.9) #Parameters to Change

print(res[0]['generated_text']) #Print Generated Text to CL/VScode

with open('gpttext.txt', 'w') as f:
    f.writelines(res[0]['generated_text'])

