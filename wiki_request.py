import time
import requests
from bs4 import BeautifulSoup
import os
import json
import re
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

ANIMAL_DIR = 'D:\\py\\animals'
PLANT_DIR = 'D:\\py\\plants'
DEPTH = 3
max_page = retain_rate = 0.07

# json_dict = {"animals": []}
json_dict = {"plants": []}

def SSL_pass(url, proxy, tout):
    try:
        return requests.get(url, verify=False, proxies=proxy, timeout=tout)
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        SSL_pass(url, proxy, tout)

if os.path.exists(PLANT_DIR):
    os.chdir(PLANT_DIR)
else:
    os.chdir('D:\\py')
    os.mkdir('plants')
    os.chdir('.\\plants')

assert os.getcwd() == PLANT_DIR

title_geter = re.compile('(?!/)(\w)+$')

f = open('wikidatas.json', 'w')

url = "https://en.wikipedia.org/wiki/Lists_of_plants"
proxies = {"http": "https://127.0.0.1:10809", 
           "https": "https://127.0.0.1:10809"}

print("start requesting...")

# get a page
animap_page = SSL_pass(url, proxy=proxies, tout=1200) 
animap_page.encoding = 'utf-8'
soup = BeautifulSoup(animap_page.text, 'html.parser')

print('have launched!!!')

cont = soup.find("div", class_="mw-parser-output")
content = cont.find_all("ul")
inners = ""

print(content)

for tes in content[:-1]:
    inners = inners + tes.text

title = title_geter.search(url)

links = []
for single_content in content:
    for link in single_content.find_all('a'):
        href = link.get('href')
        if href not in ['None', None]:
            if href.startswith("/wiki/"):
                links.append(href)
record_dict = {}
record_dict[title[0]] = {"text": inners, "link": links}
json_dict["plants"].append(record_dict)

for _ in range(DEPTH):
    current_dpeth_links = [] # links recorder in this recurrent round
    if _ == 0:
        gloal_links = [] # total URLs from parents rounds
        gloal_links.append(links)
    for sg_link in tqdm(gloal_links[_]):
        url = 'https://en.wikipedia.org' + sg_link
        page = SSL_pass(url, proxy=proxies, tout=1200)
        if not page:
            continue        
        if (np.random.rand() > retain_rate) & (_ == 2):
            continue
        page.encoding = 'utf-8'
        soup = BeautifulSoup(page.text, 'html.parser')

        cont = soup.find("div", class_="mw-parser-output")
        content = cont.find_all("p")
        inners = ""
        for tes in content[:-1]:
            inners = inners + tes.text

        title = title_geter.search(url)
        if not title:
            continue
        sub_links = [] # links in one page
        for single_content in content:
            for link in single_content.find_all('a'):
                href = link.get('href')
                if href not in ['None', None]:
                    if href.startswith("/wiki/"):
                        sub_links.append(href)
        record_dict = {}
        record_dict[title[0]] = {"text": inners, "link": sub_links}
        json_dict["plants"].append(record_dict)
        current_dpeth_links.extend(sub_links)
    gloal_links.append(current_dpeth_links)

dumpers = json.dumps(json_dict)
json.dump(dumpers, f)
f.close()