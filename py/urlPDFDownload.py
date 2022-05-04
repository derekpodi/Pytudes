#!usr/bin/env python3

#urlPDFDownload.py -- pass url at cli, download pdf on page
# $ python3 urlPDFDownload.py <{fill in pdf url here}/>


import requests
import argparse

def downloadPDF(url: str) -> None:
    response = requests.get(url)
    with open('/Users/derek/Downloads/' + url[-9:-4] + '.pdf', 'wb') as f:  #ocw specific
        f.write(response.content)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, 
                    help="input URL for pdf")
    args = parser.parse_args()
    url = args.input

    try:
        downloadPDF(url)
    except:
        print("Exception occured from input.")
   
def batch():
    #Place many url links in array, can dowload many pdf's in one run
    links = [
        #Links Here
    ]

    try:
        for url in links:
            downloadPDF(url)
    except:
        print("Exception occured from input.")

if __name__ == '__main__':
    main()
    #batch()