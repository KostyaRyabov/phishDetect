from sys import settrace
from colour import Color
from tldextract import extract as tld_extract
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urlsplit, urljoin
from numpy import array, asarray
from cv2 import INTER_AREA, GaussianBlur, resize, IMREAD_COLOR, imdecode
from pytesseract import pytesseract, image_to_string
from re import compile, finditer, MULTILINE, DOTALL, search, findall
import socket
from OpenSSL.crypto import load_certificate, FILETYPE_PEM
from whois import whois
from requests import session, exceptions
from iso639 import languages
from threading import Thread
from pickle import load
from wordninja import LanguageModel
from tkinter import END, Tk, StringVar, Entry, Button, N, S, W, E, Text, HORIZONTAL
from tkinter.ttk import Progressbar, Style

import re
from requests import get
import concurrent.futures
import lxml

from concurrent.futures._base import TimeoutError
from googletrans import Translator
from bs4 import BeautifulSoup
import ssl
from datetime import datetime
from collections import Counter

from bloomfpy import BloomFilter

pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

translator = Translator()
word_splitter = LanguageModel('data/wordlist.txt.gz')
brand_filter = load(open('data/brands.pkl', 'rb'))
words_filter = load(open('data/words.pkl', 'rb'))
phish_hints = load(open('data/phish_hints.pkl', 'rb'))
classifier = load(open('data/classifier.pkl', 'rb'))
OPR_key = open("data/OPR_key.txt").read()

reg = compile(r'\w{3,}')
http_header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
    'Content-Type': "text/html; charset=utf-8"}
progress_task = None

class KThread(Thread):
    def __init__(self, *args, **keywords):
        Thread.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        Thread.start(self)

    def __run(self):
        settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, why, arg):
        if why == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, why, arg):
        if self.killed:
            if why == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True

def run_in_thread(fn):
    def run(*k, **kw):
        global progress_task
        if progress_task:
            if progress_task.is_alive():
                progress_task.kill()

        progress_task = KThread(target=fn, args=k, kwargs=kw)
        progress_task.start()
    return run

def is_URL_accessible(url, time_out=5):
    page = None

    if not url.startswith('http'):
        url = 'http://' + url

    if urlparse(url).netloc.startswith('www.'):
       url = url.replace("www.", "", 1)

    try:
        page = get(url, timeout=time_out, headers=http_header)
    except exceptions.RequestException as err:
        return False, err
    except Exception as err:
        return False, err

    if page:
        if page.status_code == 200 and page.content not in ["b''", "b' '"]:
            return True, page
        else:
            return False, 'HTTP Status Code: {}'.format(page.status_code)
    else:
        return False, 'Invalid Input!'

def https_token(scheme):
    if scheme == 'https':
        return 0
    return 1

def count_external_redirection(page, domain):
    if len(page.history) == 0:
        return 0
    else:
        count = 0
        for i, response in enumerate(page.history):
            if domain not in urlparse(response.url).netloc.lower():
                count += 1
        return count

def random_word(word):
    if word in words_filter:
        return 0
    return 1

def random_words(url_words):
    return sum([random_word(word) for word in url_words])

def sld_in_brand(sld):
    if sld in brand_filter:
        return 1
    return 0

def average_word_length(url_words):
    if len(url_words) == 0:
        return 0
    return sum(len(word) for word in url_words) / len(url_words)

def longest_word_length(url_words):
    if len(url_words) == 0:
        return 0
    return max([len(word) for word in url_words])

def whois_registered_domain(whois_domain, domain):
    try:
        hostname = whois_domain.domain_name
        if type(hostname) == list:
            for host in hostname:
                if search(host.lower(), domain):
                    return 1
            return 0.5
        else:
            if search(hostname.lower(), domain):
                return 1
            else:
                return 0.5
    except:
        return 0

session = session()

def web_traffic(short_url):
    try:
        rank = BeautifulSoup(session.get("http://data.alexa.com/data?cli=10&dat=s&url=" + short_url, timeout=3).text, "xml").find("REACH")['RANK']

        return min(int(rank) / 10000000, 1)
    except:
        return 1

def page_rank(domain):
    url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
    try:
        request = session.get(url, headers={'API-OPR': OPR_key}, timeout=3)
        result = request.json()
        result = result['response'][0]['page_rank_decimal']
        if result:
            return result
        else:
            return 0
    except:
        return 0

def get_certificate(host, port=443, timeout=3):
    context = ssl.create_default_context()
    conn = socket.create_connection((host, port), timeout=timeout)
    sock = context.wrap_socket(conn, server_hostname=host)
    try:
        der_cert = sock.getpeercert(True)
    finally:
        sock.close()
    return ssl.DER_cert_to_PEM_cert(der_cert)

def get_cert(hostname):
    result = None
    try:
        certificate = get_certificate(hostname)
        x509 = load_certificate(FILETYPE_PEM, certificate)

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

    return result

def count_alt_names(cert):
    try:
        return len(cert[b'subjectAltName'].split(','))
    except:
        return 0

def good_netloc(netloc):
    try:
        socket.gethostbyname(netloc)
        return 1
    except:
        return 0

def urls_ratio(urls, total_urls):
    if len(total_urls) == 0:
        return 0
    else:
        return len(urls) / len(total_urls)

def ratio_List(Arr, key):
    total = len(Arr['internals']) + len(Arr['externals']) + len(Arr['null'])

    if 'embedded' in Arr:
        total += Arr['embedded']

    if total == 0:
        return 0
    elif key == 'embedded':
        return min(Arr[key] / total, 1)
    else:
        return min(len(Arr[key]) / total, 1)

def ratio_anchor(Anchor, key):
    total = len(Anchor['safe']) + len(Anchor['unsafe']) + len(Anchor['null'])

    if total == 0:
        return 0
    else:
        return len(Anchor[key]) / total

def get_html_from_js(context):
    pattern = r"([\"'`])[\s\w]*(<\s*(\w+)[^>]*>.*(<\s*\/\s*\3\s*>)?)[\s\w]*\1"
    return " ".join([r.group(2) for r in finditer(pattern, context, MULTILINE) if r.group(2) is not None])

def remove_JScomments(string):
    pattern = r"(\".*?\"|\'.*?\'|\`.*?\`)|(/\*.*?\*/|//[^\r\n]*$)"
    regex = compile(pattern, MULTILINE | DOTALL)

    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(2)

    return regex.sub(_replacer, string)

def count_io_commands(string):
    pattern = r"(\".*?\"|\'.*?\'|\`.*?\`)|" \
              r"((.(open|send)|$.(get|post|ajax|getJSON)|fetch|axios(|.(get|post|all))|getData)\s*\()"
    regex = compile(pattern, MULTILINE | DOTALL)

    count = 0

    for m in finditer(regex, string):
        if not m.group(2) and m.groups():
            count += 1

    return count

def translate_image(obj):
    try:
        resp = session.get(obj[0], stream=True, timeout=3).raw
        image = asarray(bytearray(resp.read()), dtype="uint8")
        img = imdecode(image, IMREAD_COLOR)
        img = resize(img, None, fx=0.5, fy=0.5, interpolation=INTER_AREA)
        img = GaussianBlur(img, (5, 5), 0)

        return image_to_string(img, lang=obj[1])
    except:
        return ""

def image_to_text(img, lang):
    if not img:
        return ""

    try:
        lang = languages.get(alpha2=lang).bibliographic
    except:
        return ""

    if 'eng' not in lang:
        lang = 'eng+' + lang

    docs = []

    try:
        with ThreadPoolExecutor(25) as executor:
            for r in executor.map(translate_image, [(url, lang) for url in img], timeout=3):
                if r:
                    docs.append(r)
        return "\n".join(docs)
    except TimeoutError:
        return "\n".join(docs)

def ratio_Txt(dynamic, static):
    total = len(static)

    if total:
        return min(len(dynamic) / total, 1)
    else:
        return 0

def count_phish_hints(word_raw, lang):
    if type(word_raw) == list:
        word_raw = ' '.join(word_raw)

    try:
        exp = '|'.join(list(set([item for sublist in [phish_hints[lang], phish_hints['en']] for item in sublist])))

        if exp:
            return len(findall(exp, word_raw))/len(word_raw)
        else:
            return 0
    except:
        return 0

def check_Language(text):
    global phish_hints

    size = len(text)
    if size > 10000:
        size = 10000

    try:
        language = translator.detect(str(text)[:size]).lang

        if type(language) is list:
            if 'en' in language:
                language.remove('en')
            language = language[-1]

        return language
    except:
        return 'en'

def get_domain(url):
    o = urlsplit(url)
    return o.hostname, tld_extract(url).domain, o.path, o.netloc

def extract_all_context_data(hostname, content, domain, base_url):
    Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                   "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

    Href = {'internals': [], 'externals': [], 'null': []}
    Link = {'internals': [], 'externals': [], 'null': []}
    Anchor = {'safe': [], 'unsafe': [], 'null': []}
    Img = {'internals': [], 'externals': [], 'null': []}
    Media = {'internals': [], 'externals': [], 'null': []}
    Form = {'internals': [], 'externals': [], 'null': []}
    SCRIPT = {'internals': [], 'externals': [], 'null': [], 'embedded': 0}

    CSS = {'internals': [], 'externals': [], 'null': [], 'embedded': 0}
    Favicon = {'internals': [], 'externals': [], 'null': []}

    Text = ''

    soup = BeautifulSoup(content, 'lxml')


    def collector1():
        for script in soup.find_all('script', src=True):
            url = script['src']

            if url in Null_format:
                url = 'http://' + hostname + '/' + url
                SCRIPT['null'].append(url)
                Link['null'].append(url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                SCRIPT['internals'].append(url)
                Link['internals'].append(url)
            else:
                SCRIPT['externals'].append(url)
                Link['externals'].append(url)
    def collector2():
        for href in soup.find_all('a', href=True):
            url = href['href']

            if "#" in url or "javascript" in url or "mailto" in url:
                Anchor['unsafe'].append('http://' + hostname + '/' + url)

            if url in Null_format:
                Href['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Href['internals'].append(url)
            else:
                Href['externals'].append(url)
                Anchor['safe'].append(url)
    def collector3():
        for img in soup.find_all('img', src=True):
            url = img['src']

            if url in Null_format:
                url = 'http://' + hostname + '/' + url
                Media['null'].append(url)
                Img['null'].append(url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Media['internals'].append(url)
                Img['internals'].append(url)
            else:
                Media['externals'].append(url)
                Img['externals'].append(url)
    def collector4():
        for audio in soup.find_all('audio', src=True):
            url = audio['src']

            if url in Null_format:
                Media['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Media['internals'].append(url)
            else:
                Media['externals'].append(url)
    def collector5():
        for embed in soup.find_all('embed', src=True):
            url = embed['src']

            if url in Null_format:
                Media['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Media['internals'].append(url)
            else:
                Media['externals'].append(url)
    def collector6():
        for i_frame in soup.find_all('iframe', src=True):
            url = i_frame['src']

            if url in Null_format:
                Media['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Media['internals'].append(url)
            else:
                Media['externals'].append(url)
    def collector7():
        for link in soup.findAll('link', href=True):
            url = link['href']

            if url in Null_format:
                Link['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Link['internals'].append(url)
            else:
                Link['externals'].append(url)
    def collector8():
        for form in soup.findAll('form', action=True):
            url = form['action']

            if url in Null_format or url == 'about:best_nn':
                Form['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Form['internals'].append(url)
            else:
                Form['externals'].append(url)
    def collector9():
        for link in soup.find_all('link', rel='stylesheet'):
            url = link['href']

            if url in Null_format:
                CSS['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                CSS['internals'].append(url)
            else:
                CSS['externals'].append(url)

        CSS['embedded'] = len([css for css in soup.find_all('style', type='text/css') if len(css.contents) > 0])
    def collector10():
        for head in soup.find_all('head'):
            for head.link in soup.find_all('link', href=True):
                url = head.link['href']

                if url in Null_format:
                    Favicon['null'].append('http://' + hostname + '/' + url)
                    continue

                url = urljoin(base_url, url)

                if domain in urlparse(url).netloc:
                    Favicon['internals'].append(url)
                else:
                    Favicon['externals'].append(url)

            for head.link in soup.findAll('link', {'href': True, 'rel': True}):
                isicon = False
                if isinstance(head.link['rel'], list):
                    for e_rel in head.link['rel']:
                        if e_rel.endswith('icon'):
                            isicon = True
                            break
                else:
                    if head.link['rel'].endswith('icon'):
                        isicon = True
                        break

                if isicon:
                    url = head.link['href']

                    if url in Null_format:
                        Favicon['null'].append('http://' + hostname + '/' + url)
                        continue

                    url = urljoin(base_url, url)

                    if domain in urlparse(url).netloc:
                        Favicon['internals'].append(url)
                    else:
                        Favicon['externals'].append(url)
    def merge_scripts(script_lnks):
        docs = []

        def load_script(url):
            state, request = is_URL_accessible(url, 2)

            if state:
                request.encoding = 'utf-8'
                docs.append(request.text)

        try:
            with ThreadPoolExecutor(25) as executor:
                res = executor.map(load_script, script_lnks, timeout=3)

                for r in res:
                    if r:
                        docs.append(r)
            return "\n".join(docs)
        except TimeoutError:
            return "\n".join(docs)

    with ThreadPoolExecutor(13) as e:
        e.submit(collector1)
        e.submit(collector2)
        e.submit(collector3)
        e.submit(collector4)
        e.submit(collector5)
        e.submit(collector6)
        e.submit(collector7)
        e.submit(collector8)
        e.submit(collector9)
        e.submit(collector10)
        internals_script_doc = e.submit(merge_scripts, SCRIPT['internals']).result()
        externals_script_doc = e.submit(merge_scripts, SCRIPT['externals']).result()
        Text = e.submit(soup.get_text).result().lower()

    try:
        internals_script_doc = ' '.join(
            [internals_script_doc] + [script.contents[0] for script in soup.find_all('script', src=False) if
                                      len(script.contents) > 0])

        SCRIPT['embedded'] = len(
            [script.contents[0] for script in soup.find_all('script', src=False) if len(script.contents) > 0])
    except:
        pass

    io_count = len(soup.find_all('textarea')) + len(soup.find_all('input', type=None))
    for io in soup.find_all('input', type=True):
        if io['type'] == 'text' or io['type'] == 'password' or io['type'] == 'search':
            io_count += 1

    return Href, Link, Anchor, Media, Img, Form, CSS, Favicon, Text, internals_script_doc.lower(), externals_script_doc.lower(), io_count

def extract_onlyText(content):
    return BeautifulSoup(content, 'lxml').get_text().lower()

def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)|'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '[0-9a-fA-F]{7}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tr\.im|is\.gd|cli\.gs|yfrog\.com|'
                      'migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|'
                      'ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|'
                      'wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|'
                      'tinyurl\.com|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|'
                      'yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|'
                      'v\.gd|link\.zip\.net|sh\.st|doma\.in|urlhum\.com|flipmsg\.com|dik\.si|cutt\.ly|1t\.ie|inlnk\.ru|'
                      'link\.ly|nlsn\.cf|neya\.link|chl\.li|tinu\.be|Tiky\.cc|sho\.pw|LNKS\.ES|bulkurlshortener\.com|'
                      'btfy\.io|cmpct\.io|zee\.gl|rb\.gy|short\.io|smarturl\.it|hyperurl\.co|soo\.gd|t\.ly|tinycc.com|'
                      'shorturl\.at|polr\.me|hypr\.ink|zpr\.io|goo-gl\.ru|clc\.am|bitly\.is|lnnk\.in|vk\.cc|clck\.ru|'
                      't.co|gestyy\.com|rb\.gy',
                      url)
    if match:
        return 1
    else:
        return 0

def ratio_digits(url):
    return len(re.sub("[^0-9]", "", url)) / len(url)

def count_digits(url):
    return len(re.sub("[^0-9]", "", url))

def char_repeat(words_raw):
    if words_raw:
        count = 0

        for word in words_raw:
            if word:
                for i in range(len(word)-1):
                    if word[i] == word[i+1]:
                        count += 1/len(word)
        return count / len(words_raw)
    else:
        return 0

def brand_in_path(words_raw_path):
    for word in words_raw_path:
        if word in brand_filter:
            return 1
    return 0

def count_subdomain(netloc):
    return len(re.findall("\.", netloc))

def compression_ratio(request):
    try:
        compressed_length = int(request.headers['content-length'])
        decompressed_length = len(request.content)
        return compressed_length / decompressed_length
    except:
        return 1

def fetch(url):
    try:
        return get(url, timeout=3)
    except:
        return None

def get_reqs_data(url_list):
    return_value = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            for req in executor.map(fetch, url_list, timeout=3):
                return_value.append(req)
    except:
        pass
    return return_value

from collections import defaultdict
import threading

def segment(*text):
    return word_splitter.split(''.join(text).lower())

def word_ratio(*Text_words):
    Text_words = [item for sublist in Text_words for item in sublist]

    if Text_words:
        return len(Counter(Text_words)) / len(Text_words)
    else:
        return 0

class Manager:
    def url_stats(self, url, r_url, request):
        self.result[1] = having_ip_address(url)
        self.result[2] = shortening_service(url)
        self.result[4] = len(r_url)
        self.result[5] = r_url.count('@')
        self.result[6] = r_url.count(';')
        self.result[7] = r_url.count('&')
        self.result[8] = r_url.count('/') - 2
        self.result[9] = r_url.count('=')
        self.result[10] = r_url.count('%')
        self.result[11] = r_url.count('-')
        self.result[12] = r_url.count('.')
        self.result[13] = r_url.count('~')
        self.result[15] = ratio_digits(r_url)
        self.result[16] = count_digits(r_url)
        self.result[18] = len(request.history)
        self.result[30] = compression_ratio(request)
    def domain_info(self, r_url):
        hostname, second_level_domain, path, netloc = get_domain(r_url)

        self.set('hostname', hostname)
        self.set('second_level_domain', second_level_domain)
        self.set('netloc', netloc)
    def get_url_parts(self, extracted_domain, r_url):
        self.set('domain', extracted_domain.domain + '.' + extracted_domain.suffix)

        tmp = r_url[r_url.find(extracted_domain.suffix):len(r_url)]

        pth = tmp.partition("/")[2]
        self.set('pth', pth)

        cutted_url = extracted_domain.domain + extracted_domain.subdomain
        self.set('cutted_url', cutted_url)
        self.set('cutted_url2', cutted_url + pth)
    def update_url_parts(self, url_words, parsed):
        self.result[17] = len(url_words)
        self.set('scheme', parsed.scheme)
    def url_stats2(self, scheme, cutted_url2):
        self.result[14] = https_token(scheme)
        self.result[25] = cutted_url2.count('www')
        self.result[26] = cutted_url2.count('com')
    def url_lens(self, url_words):
        self.result[27] = average_word_length(url_words)
        self.result[28] = longest_word_length(url_words)
    def sContext_grabber(self, hostname, content, domain, r_url):
        (Href, Link, Anchor, Media, Img, Form, CSS, Favicon, Text, internals_script_doc, externals_script_doc,
         io_count) = extract_all_context_data(hostname, content, domain, r_url)

        iUrl_s = Href['internals'] + Link['internals'] + Media['internals'] + Form['internals']
        eUrl_s = Href['externals'] + Link['externals'] + Media['externals'] + Form['externals']
        nUrl_s = Href['null'] + Link['null'] + Media['null'] + Form['null']

        self.set('Text', Text)

        self.set('internals_script_doc', internals_script_doc)
        self.set('externals_script_doc', externals_script_doc)

        self.set('Img_internals', Img['internals'])
        self.set('Img_externals', Img['externals'])

        self.result[31] = io_count
        self.result[32] = len(iUrl_s) + len(eUrl_s)
        self.result[33] = urls_ratio(iUrl_s, iUrl_s + eUrl_s + nUrl_s)
        self.result[34] = urls_ratio(eUrl_s, iUrl_s + eUrl_s + nUrl_s)
        self.result[35] = urls_ratio(nUrl_s, iUrl_s + eUrl_s + nUrl_s)
        self.result[36] = ratio_List(CSS, 'embedded')
        self.result[37] = ratio_List(Img, 'internals')
        self.result[38] = ratio_List(Favicon, 'internals')
        self.result[39] = ratio_List(Media, 'externals')
        self.result[40] = ratio_anchor(Anchor, 'unsafe')
        self.result[41] = ratio_anchor(Anchor, 'safe')
        self.result[42] = ratio_List(Link, 'internals')
        self.result[43] = ratio_List(Link, 'externals')
    def cert_stats(self, whois_domain, domain, cert):
        self.result[49] = whois_registered_domain(whois_domain, domain)
        self.result[52] = count_alt_names(cert)
    def Text_stats(self, iImgTxt_words, eImgTxt_words, sContent_words):
        self.result[46] = ratio_Txt(iImgTxt_words + eImgTxt_words, sContent_words)

    def __init__(self):
        self.event = threading.Event()

        self.result = [None] * 53
        self.values = {}
        self.index = defaultdict(set)
        self.tasks = []

        self.tasks.append([self.url_stats, ['url', 'r_url', 'request'], -1])
        self.tasks.append([self.domain_info, ['r_url'], -1])
        self.tasks.append([self.get_url_parts, ['extracted_domain', 'r_url'], -1])
        self.tasks.append([tld_extract, ['r_url'], 'extracted_domain'])
        self.tasks.append([good_netloc, ['netloc'], 3])
        self.tasks.append([count_external_redirection, ['request', 'domain'], 19])
        self.tasks.append([sld_in_brand, ['second_level_domain'], 23])
        self.tasks.append([count_subdomain, ['netloc'], 29])
        self.tasks.append([page_rank, ['domain'], 51])
        self.tasks.append([segment, ['pth'], 'words_raw_path'])
        self.tasks.append([segment, ['cutted_url'], 'words_raw_host'])
        self.tasks.append([segment, ['cutted_url', 'pth'], 'url_words'])
        self.tasks.append([urlparse, ['domain'], 'parsed'])
        self.tasks.append([self.update_url_parts, ['url_words', 'parsed'], -1])
        self.tasks.append([random_words, ['url_words'], 20])
        self.tasks.append([char_repeat, ['words_raw_host'], 21])
        self.tasks.append([char_repeat, ['words_raw_path'], 22])
        self.tasks.append([brand_in_path, ['words_raw_path'], 24])
        self.tasks.append([self.url_stats2, ['scheme', 'cutted_url2'], -1])
        self.tasks.append([self.url_lens, ['url_words'], -1])
        self.tasks.append([web_traffic, ['r_url'], 50])
        self.tasks.append([whois, ['domain'], 'whois_domain'])
        self.tasks.append([get_cert, ['domain'], 'cert'])
        self.tasks.append([self.sContext_grabber, ['hostname', 'content', 'domain', 'r_url'], -1])
        self.tasks.append([count_io_commands, ['internals_script_doc'], 47])
        self.tasks.append([count_io_commands, ['externals_script_doc'], 48])
        self.tasks.append([self.cert_stats, ['whois_domain', 'domain', 'cert'], -1])
        self.tasks.append([reg.findall, ['Text'], 'sContent_words'])
        self.tasks.append([check_Language, ['Text'], 'lang'])
        self.tasks.append([remove_JScomments, ['internals_script_doc'], 'js_di'])
        self.tasks.append([remove_JScomments, ['externals_script_doc'], 'js_de'])
        self.tasks.append([len, ['sContent_words'], 45])
        self.tasks.append([count_phish_hints, ['Text', 'lang'], 44])
        self.tasks.append([image_to_text, ['Img_internals', 'lang'], 'internals_img_txt'])
        self.tasks.append([image_to_text, ['Img_externals', 'lang'], 'externals_img_txt'])

        self.tasks.append([get_html_from_js, ['js_di'], 'content_di'])
        self.tasks.append([get_html_from_js, ['js_de'], 'content_de'])

        self.tasks.append([extract_onlyText, ['content_di'], 'Text_di'])
        self.tasks.append([extract_onlyText, ['content_de'], 'Text_de'])
        self.tasks.append([reg.findall, ['Text_di'], 'diContent_words'])
        self.tasks.append([self.Text_stats, ['iImgTxt_words', 'eImgTxt_words', 'sContent_words'], -1])
        self.tasks.append([reg.findall, ['Text_de'], 'deContent_words'])
        self.tasks.append(
            [word_ratio, ['iImgTxt_words', 'eImgTxt_words', 'sContent_words', 'diContent_words', 'deContent_words'], 0])
        self.tasks.append([reg.findall, ['internals_img_txt'], 'iImgTxt_words'])
        self.tasks.append([reg.findall, ['externals_img_txt'], 'eImgTxt_words'])

        for task_idx, options in enumerate(self.tasks):
            for requires in options[1]:
                self.index[requires] |= {task_idx}
            options += [options[1].copy()]

    def set(self, key, val):
        self.values[key] = val

        for t in self.index[key]:
            (fun, dependencies, prob, required) = self.tasks[t]

            required.remove(key)
            if not required:
                def t_fun(f, prob, d):
                    f_res = f(*d)

                    if type(prob) == str:
                        self.set(prob, f_res)
                    elif prob >= 0:
                        self.result[prob] = f_res

                    global progress, p_v
                    p_v += 1
                    progress['value'] = p_v

                    if None not in self.result:
                        self.event.set()

                thread = threading.Thread(target=t_fun, args=(fun, prob, [self.values[d] for d in dependencies]))
                thread.start()

    def get_result(self):
        self.event.wait()
        return self.result


def extract_features(url):
    try:
        (state, request) = is_URL_accessible(url, 3)

        if state:
            request.encoding = 'utf-8'

            manager = Manager()
            manager.set('request', request)
            manager.set('url', url)
            manager.set('r_url', request.url)
            manager.set('content', request.text.lower())

            return manager.get_result()
        return request
    except Exception as ex:
        print(ex)
        return ex

d = [
    1.0,1.0,1.0,1.0,1648.0,5.0,12.0,17.0,21.0,18.0,174.0,30.0,47.0,4.0,1.0,0.9680722891566264,1607.0,1072.0,9.0,6.0,640.0,0.6666666666666666,0.6666666666666666,1.0,1.0,4.0,6.0,30.0,30.0,11.0,7.634838196231642,969.0,50791.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.2,1385905.0,1.0,422645.0,285064.0,1.0,0.9999999,10.0,804.0
]

if __name__ == "__main__":
    @run_in_thread
    def check_site():
        result.configure(background='white')
        result.configure(state='normal')
        result.delete(1.0, END)
        result.configure(state='disabled')

        global p_v, progress
        p_v = 1
        progress['value'] = p_v

        data = extract_features(url.get().strip())

        if type(data) is list:
            print('1:[{}]'.format(len(data)))
            data = array([max(min(data[i]/d[i], 1), 0) for i in range(53)]).reshape((1, -1)) * 0.9998 + 0.0001
            print('1:[{}]'.format(data))
            res = classifier.predict_proba(data).tolist()[0][-1]

            result.configure(state='normal')
            result.configure(background=Color(hsl=(0.2778*(1-res), 1, 0.5)).get_hex_l())

            if res < 0.5:
                result.insert(END, "\nЭто легитимный сайт!".format((1-res)*100), 'tag-center')
            else:
                result.insert(END, "\nЭто фишинговый сайт!".format(res * 100), 'tag-center')

            result.configure(state='disabled')
        else:
            result.configure(state='normal')
            result.insert(END, "ERROR: {}".format(data))
            result.configure(state='disabled')
        progress['value'] = 47

    window = Tk()
    window.title("phishDetect")
    window.resizable(0, 0)

    url = StringVar()

    textArea = Entry(textvariable=url, width=80, exportselection=0)
    textArea.grid(column=0, row=0, sticky=N+S+W+E)

    btn = Button(window, text="check", command=check_site)
    btn.grid(column=1, row=0, sticky=N+S+W+E)

    s = Style()
    s.theme_use("default")
    s.configure("TProgressbar", thickness=2)

    progress = Progressbar(
        window,
        orient=HORIZONTAL,
        maximum=47,
        length=100,
        mode='determinate',
        style="TProgressbar"
    )
    progress.grid(column=0, row=1, columnspan=2,  sticky=N + S + W + E)

    result = Text(
        window,
        height=3,
        width=80,
        state='disabled'
    )
    result.tag_configure('tag-center', justify='center')

    result.grid(column=0, row=2, columnspan=2)

    window.mainloop()

    if progress_task:
        if progress_task.is_alive():
            progress_task.kill()