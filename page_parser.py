

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import time
import requests
import re
from collections import Counter
import selenium
from selenium import webdriver
import logging
import numpy as np


def parseLink(link):
    try:
        linkInfo = re.split('\W+',link.replace('https://','').replace('http://','').replace('www.','').replace('.com',''))
        linkInfo = ' '.join(list(filter(lambda x:x and len(x)>0, linkInfo)))
        return linkInfo
    except:
        return ''

def getScreadPageContent(req):

    soup = BeautifulSoup(req,'html.parser')
    try:
        script_info = soup.find_all('script')
        script_info = list(filter(lambda x:x and x.get('src') and len(x.get('src'))>1, script_info))
        script = list(map(lambda x:x.get('src'),script_info))
        script = '. '.join(list(map(lambda x:parseLink(x),script)))
    except:
        script = ''
    
    # get all the img
    try:
        img_info = soup.find_all('img')
        imgs_info = list(filter(lambda x:x and x.get('src') and len(x.get('src'))>1, img_info))
        imgs = list(map(lambda x:x.get('src'),img_info))
        # imgs = list(filter(lambda x:x is not None and len(x)>0,imgs))
        imgs = '. '.join(list(map(lambda x:parseLink(x),imgs)))
    except:
        imgs = ''
    
    # get all the links
    try:
        url_info = soup.find_all('a')
        url_info = list(filter(lambda x:x and x.get('href') and len(x.get('href'))>1, url_info))
        urls = list(map(lambda x:x.get('href'),url_info))
        # urls = list(filter(lambda x:x is not None and len(x)>0,urls))
        urls = '. '.join(list(map(lambda x:parseLink(x),urls)))
    except:
        ulrs = ''
    
    for scripts in soup(["script", "style"]):
        scripts.extract()
    # get text
    content = soup.get_text().split('\n')
    content = list(map(lambda x:' '.join(re.sub(r'[^a-zA-Z ]','',x).split()).lower(),content))
    content = '. '.join(list(filter(lambda x:len(x)<50 and len(x)>1,content)))
    
    # get all elemnents' class names
    try:
        all_tag_elems = soup.find('body').find_all()
        all_tag_elems = list(filter(lambda x:x and x.name and x.get('class'), all_tag_elems))
        all_tags = list(map(lambda x:(x.name,' '.join(x.get('class'))),all_tag_elems))
        
        freqs = Counter(all_tags).items()
        all_tags = list(filter(lambda x:x[1]>1,freqs))
        
        tags = '. '.join(list(map(lambda x:x[0][0],all_tags)))
        class_values = '. '.join(list(map(lambda x:x[0][1],all_tags)))
    except:
        tags = class_values = ''
    return [script, imgs, urls, content, tags, class_values]
