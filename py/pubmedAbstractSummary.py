#!usr/bin/env python3

#pubmedAbstractSummary.py -- Command Line Program that takes input of PubMed URL. Pulls abstract text, and summarizes it
#Run in Virtual Enviroment:
# $ python3 pubmedAbstractSummary.py <https://pubmed.ncbi.nlm.nih.gov/{fill in article id here}/>

import argparse
import requests
from bs4 import BeautifulSoup
from transformers import pipeline


def extractAbstract(url: str) -> str:
    req = requests.get(url).text
    soup = BeautifulSoup(req, 'html.parser')
    res = soup.find("div", class_="abstract-content selected").p.text.strip()
    return res


def sumAbstract(abstract:str) -> str:
    classifier = pipeline("summarization", model='sshleifer/distilbart-cnn-12-6')
    res = classifier(abstract)
    ret = res[0]['summary_text']
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, 
                    help="input PubMed URL for summarization")
    args = parser.parse_args()
    url = args.input

    try:
        abstract = extractAbstract(url)
        print("--------------------------------")
        print("Abstract:")
        print(abstract)
        print("--------------------------------")
        summary = sumAbstract(abstract)
        print("Summary:")
        print(summary)
        print("--------------------------------")
    except:
        print("Exception occured from input.")
   

if __name__ == '__main__':
    main()