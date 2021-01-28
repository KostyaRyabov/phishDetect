from datetime import datetime
from bs4 import BeautifulSoup
import requests
import whois
import time
from tools import benchmark
import re


OPR_key = open("OPR_key.txt").read()


#################################################################################################################################
#               Domain registration age
#################################################################################################################################

@benchmark
def domain_registration_length(domain):
    try:
        res = whois.whois(domain)
        expiration_date = res.expiration_date
        today = time.strftime('%Y-%m-%d')
        today = datetime.strptime(today, '%Y-%m-%d')
        # Some domains do not have expiration dates. The application should not raise an error if this is the case.
        if expiration_date:
            if type(expiration_date) == list:
                expiration_date = min(expiration_date)
            return abs((expiration_date - today).days)
        else:
            return 0
    except:
        return -1


#################################################################################################################################
#               Domain recognized by WHOIS
#################################################################################################################################

@benchmark
def whois_registered_domain(domain):
    try:
        hostname = whois.whois(domain).domain_name
        if type(hostname) == list:
            for host in hostname:
                if re.search(host.lower(), domain):
                    return 0
            return 1
        else:
            if re.search(hostname.lower(), domain):
                return 0
            else:
                return 1
    except:
        return 1


#################################################################################################################################
#               Unable to get web traffic (Page Rank)
#################################################################################################################################
import sys, lxml

@benchmark
def web_traffic(short_url):
    try:
        rank = BeautifulSoup(requests("http://data.alexa.com/data?cli=10&dat=s&url=" + short_url).text,
                             "xml").find("REACH")['RANK']
    except:
        return sys.maxint
    return int(rank)


#################################################################################################################################
#               Domain age of a url
#################################################################################################################################

import json

@benchmark
def domain_age(domain):
    url = domain.split("//")[-1].split("/")[0].split('?')[0]
    show = "https://input.payapi.io/v1/api/fraud/domain/age/" + url
    r = requests.get(show)

    if r.status_code == 200:
        data = r.text
        jsonToPython = json.loads(data)
        result = jsonToPython['result']
        if result == None:
            return -2
        else:
            return result
    else:
        return -1


#################################################################################################################################
#               Google index
#################################################################################################################################


from urllib.parse import urlencode

@benchmark
def google_index(url):
    # time.sleep(.6)
    user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'
    headers = {'User-Agent': user_agent}
    query = {'q': 'site:' + url}
    google = "https://www.google.com/search?" + urlencode(query)
    data = requests.get(google, headers=headers)
    data.encoding = 'ISO-8859-1'
    soup = BeautifulSoup(str(data.content), "html.parser")
    try:
        if 'Our systems have detected unusual traffic from your computer network.' in str(soup):
            return -1
        check = soup.find(id="rso").find("div").find("div").find("a")
        if check and check['href']:
            return 1
        else:
            return 0

    except AttributeError:
        return 0


#################################################################################################################################
#               Page Rank from OPR
#################################################################################################################################

@benchmark
def page_rank(domain):
    url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
    try:
        request = requests.get(url, headers={'API-OPR': OPR_key})
        result = request.json()
        result = result['response'][0]['page_rank_integer']
        if result:
            return result
        else:
            return 0
    except:
        return 0