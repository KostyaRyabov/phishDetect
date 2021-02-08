from datetime import datetime
from bs4 import BeautifulSoup
import requests
import whois
import time
from tools import benchmark
import re


OPR_key = open("OPR_key.txt").read()


########################################################################################################################
#               Domain registration age
########################################################################################################################

@benchmark(2)
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


########################################################################################################################
#               Domain recognized by WHOIS
########################################################################################################################

@benchmark(2)
def whois_registered_domain(domain):
    try:
        hostname = whois.whois(domain).domain_name
        if type(hostname) == list:
            for host in hostname:
                if re.search(host.lower(), domain):
                    return 1
            return 0
        else:
            if re.search(hostname.lower(), domain):
                return 1
            else:
                return 0
    except:
        return -1


########################################################################################################################
#               Unable to get web traffic (Page Rank)
########################################################################################################################


import sys, lxml

session = requests.session()

@benchmark(2)
def web_traffic(short_url):
    try:
        rank = BeautifulSoup(session.get("http://data.alexa.com/data?cli=10&dat=s&url=" + short_url, timeout=2).text,
                             "xml").find("REACH")['RANK']
    except:
        return -1

    # return min(int(rank), 10000000)
    return int(rank)


########################################################################################################################
#               Google index
########################################################################################################################


from urllib.parse import urlencode

@benchmark(2)
def google_index(url):
    query = {'q': 'site:' + url}
    google = "https://www.google.com/search?" + urlencode(query)
    try:
        data = session.get(google,timeout=2)
        soup = BeautifulSoup(data.content, "html.parser")

        if 'Our systems have detected unusual traffic from your computer network.' in str(soup):
            return -1
        check = soup.find(id="rso").find("div").find("div").find("div").find("div").find("a")
        if check and check['href']:
            return 1
        else:
            return 0

    except AttributeError:
        return -1


########################################################################################################################
#               Page Rank from OPR
########################################################################################################################


# def http_build_query(params, topkey=''):
#     from urllib.parse import quote
#
#     if len(params) == 0:
#         return ""
#
#     result = ""
#
#     # is a dictionary?
#     if type(params) is dict:
#         for key in params.keys():
#             newkey = quote(key)
#             if topkey != '':
#                 newkey = topkey + quote('[' + key + ']')
#
#             if type(params[key]) is dict:
#                 result += http_build_query(params[key], newkey)
#
#             elif type(params[key]) is list:
#                 i = 0
#                 for val in params[key]:
#                     result += newkey + quote('[' + str(i) + ']') + "=" + quote(str(val)) + "&"
#                     i = i + 1
#
#             # boolean should have special treatment as well
#             elif type(params[key]) is bool:
#                 result += newkey + "=" + quote(str(int(params[key]))) + "&"
#
#             # assume string (integers and floats work well)
#             else:
#                 result += newkey + "=" + quote(str(params[key])) + "&"
#
#     # remove the last '&'
#     if (result) and (topkey == '') and (result[-1] == '&'):
#         result = result[:-1]
#
#     return result
#
#
# @benchmark
# def page_rank(domains):
#     url = 'https://openpagerank.com/api/v1.0/getPageRank?' + http_build_query({"domains": domains})
#     try:
#         request = session.get(url, headers={'API-OPR': OPR_key})
#         result = request.json()
#         result = [record['page_rank_decimal'] for record in result['response']]
#         return result
#     except:
#         return None

@benchmark(2)
def page_rank(domain):
    url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
    try:
        request = session.get(url, headers={'API-OPR': OPR_key}, timeout=2)
        result = request.json()
        result = result['response'][0]['page_rank_integer']
        if result:
            return result
        else:
            return 0
    except:
        return -1


########################################################################################################################
#               Google search by keywords
########################################################################################################################


from googlesearch import search

@benchmark(2)
def compare_search2url(url, domain, keywords):
    try:
        res = search("http://{}={}".format(domain, '+'.join(keywords)), 10)

        for r_url in res:
            if r_url == url:
                return 1
        return 0
    except:
        return -1


########################################################################################################################
#               Certificate information
########################################################################################################################


import ssl
import socket
import OpenSSL
from datetime import datetime


def get_certificate(host, port=443, timeout=10):
    context = ssl.create_default_context()
    conn = socket.create_connection((host, port))
    sock = context.wrap_socket(conn, server_hostname=host)
    sock.settimeout(timeout)
    try:
        der_cert = sock.getpeercert(True)
    except:
        pass
    finally:
        sock.close()
    return ssl.DER_cert_to_PEM_cert(der_cert)


import threading

lock_obj = threading.Lock()

@benchmark(3)
def get_cert(hostname):
    lock_obj.acquire()
    result = None
    try:
        certificate = get_certificate(hostname)
        x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, certificate)

        result = {
            'subject': dict(x509.get_subject().get_components()),
            'issuer': dict(x509.get_issuer().get_components()),
            'serialNumber': x509.get_serial_number(),
            'version': x509.get_version(),
            'notBefore': datetime.strptime(x509.get_notBefore().decode('ascii'), '%Y%m%d%H%M%SZ'),
            'notAfter': datetime.strptime(x509.get_notAfter().decode('ascii'), '%Y%m%d%H%M%SZ'),
        }

        extensions = (x509.get_extension(i) for i in range(x509.get_extension_count()))
        extension_data = {e.get_short_name(): str(e) for e in extensions}
        result.update(extension_data)
    except:
        pass
    finally:
        lock_obj.release()

    return result


@benchmark(2)
def count_alt_names(cert):
    try:
        return len(cert[b'subjectAltName'].split(','))
    except:
        return -1

@benchmark(2)
def valid_cert_period(cert):
    try:
        return (cert['notAfter'] - cert['notBefore']).days
    except:
        return -1

@benchmark(2)
def remainder_valid_cert(cert):
    try:
        period = cert['notAfter'] - cert['notBefore']

        today = time.strftime('%Y-%m-%d')
        today = datetime.strptime(today, '%Y-%m-%d')

        passed_time = today - cert['notBefore']

        return max(0, min(1, passed_time / period))
    except:
        return -1
