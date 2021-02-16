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

@benchmark(10)
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

@benchmark(10)
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

@benchmark(12)
def web_traffic(short_url):
    try:
        rank = BeautifulSoup(session.get("http://data.alexa.com/data?cli=10&dat=s&url=" + short_url, timeout=10).text,
                             "xml").find("REACH")['RANK']
    except:
        return -1

    # return min(int(rank), 10000000)
    return int(rank)


@benchmark(10)
def page_rank(domain):
    url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
    try:
        request = session.get(url, headers={'API-OPR': OPR_key}, timeout=7)
        result = request.json()
        result = result['response'][0]['page_rank_integer']
        if result:
            return result
        else:
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
    finally:
        sock.close()
    return ssl.DER_cert_to_PEM_cert(der_cert)


import threading

lock_obj = threading.Lock()

@benchmark(20)
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


@benchmark(3)
def count_alt_names(cert):
    try:
        return len(cert[b'subjectAltName'].split(','))
    except:
        return -1

@benchmark(3)
def valid_cert_period(cert):
    try:
        return (cert['notAfter'] - cert['notBefore']).days
    except:
        return -1

@benchmark(3)
def remainder_valid_cert(cert):
    try:
        period = cert['notAfter'] - cert['notBefore']

        today = time.strftime('%Y-%m-%d')
        today = datetime.strptime(today, '%Y-%m-%d')

        passed_time = today - cert['notBefore']

        return max(0, min(1, passed_time / period))
    except:
        return -1


########################################################################################################################
#               DNS record
########################################################################################################################


@benchmark(7)
def good_netloc(netloc):
    try:
        socket.gethostbyname(netloc)
        return 1
    except:
        return 0