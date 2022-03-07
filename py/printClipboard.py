#!/usr/bin/python3
#Prints summarization of text copied to the clipboard or via OCR

#pair with OCR screenshot program here: https://github.com/schappim/macOCR
    #Use to get draggable screenshot summary of text   
    # $ ocr | python3 printClipboard.py


import sys
from transformers import pipeline

classifier = pipeline("summarization", model='sshleifer/distilbart-cnn-12-6')

original_text = str(sys.stdin.read())

res = classifier(original_text)

print(res[0]['summary_text'])