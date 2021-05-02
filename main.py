import os
from operator import index

import pandas
import tldextract
import concurrent.futures
from urllib.parse import urlparse, urlsplit, urljoin
from nltk.tokenize import RegexpTokenizer
import wordsegment
from nltk.corpus import stopwords, brown
import numpy as np
from googletrans import Translator
import cv2
import pytesseract
from bs4 import BeautifulSoup
import re
import ssl
import socket
import OpenSSL
from datetime import datetime
import whois
import Levenshtein
from collections import Counter
import requests
from iso639 import languages
import threading
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
from tensorflow import keras

import tkinter as tk
from tkinter.ttk import Progressbar, Style
from tkinter import simpledialog

import xgboost as xgb


p_v = 0


timer = {}


def indicate(func):
    def wrapper(*args, **kwargs):
        global progress, p_v, timer
        start = time()
        res = func(*args, **kwargs)
        end = time()
        timer[func.__name__] = end - start
        p_v += 1
        progress['value'] = p_v
        return res

    return wrapper


progress_task = None


def run_in_thread(fn):
    def run(*k, **kw):
        global progress_task
        if progress_task:
            if progress_task.is_alive():
                progress_task.join(0)
                print('terminated')

        progress_task = threading.Thread(target=fn, args=k, kwargs=kw)
        progress_task.start()
    return run


pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'


def load_phishHints():
    hints_dir = "data/phish_hints/"
    file_list = os.listdir(hints_dir)

    if file_list:
        return {leng[0:2]: pandas.read_csv(hints_dir + leng, header=None)[0].tolist() for leng in file_list}
    else:
        hints = {'en': [
            'login',
            'logon',
            'sign',
            'account',
            'authorization',
            'registration',
            'user',
            'password',
            'pay',
            'name',
            'profile',
            'mail',
            'pass',
            'reg',
            'log',
            'auth',
            'psw',
            'nickname',
            'enter',
            'bank',
            'card',
            'pincode',
            'phone',
            'key',
            'visa',
            'cvv',
            'cvp',
            'cvc',
            'ccv'
        ]
        }

        data = pandas.DataFrame(hints)
        filename = "data/phish_hints/en.csv"
        data.to_csv(filename, index=False, header=False)

        return hints


translator = Translator()
phish_hints = load_phishHints()


headers = [
    'коэффициент уникальности всех слов',
    'хороший netloc',
    'длина url',
    'кол-во @ в url',
    'кол-во ; в url',
    'кол-во = в url',
    'кол-во % в url',
    'кол-во - в url',
    'кол-во . в url',
    'кол-во : в url',
    'https',
    'кол-во фишинговых слов в url',
    'кол-во перенаправлений на сайт',
    'кол-во перенаправлений на другие домены',
    'случайный домен',
    'кол-во случайных слов в url',
    'кол-во случайных слов в хосте url',
    'домен в брендах',
    'кол-во www в url',
    'кол-во слов в url',
    'средняя длина слова в url',
    'максимальная длина слова в url',
    'кол-во всех ссылок в основном контексте страницы',
    'соотношение внутренних ссылок на сайты со всеми в основном контексте страницы',
    'соотношение пустых ссылок на сайты со всеми в основном контексте страницы',
    'соотношение внутренних скриптов со всеми в основном контексте страницы',
    'соотношение внутренних изображений со всеми в основном контексте страницы',
    'соотношение внешних медиа со всеми в основном контексте страницы',
    'соотношение небезопасных якорей со всеми в основном контексте страницы',
    'соотношение безопасных якорей со всеми в основном контексте страницы',
    'кол-во слов в тексте в основном контексте страницы',
    'соотношение текста со всех изображений с основным текстом в основном контексте страницы',
    'соотношение текста внутренних изображений с основным текстом в основном контексте страницы',
    'кол-во операций ввода/вывода во внутренне добавляемом коде страницы',
    'кол-во операций ввода/вывода во внешне добавляемом коде страницы',
    'домен зарегестрирован',
    'рейтинг по Alexa',
    'рейтинг по openpagerank',
    'срок действия сертификата',
    'кол-во альтернативных имен в сертификате'
]


brand_list = [brand.split('.')[0] for brand in
                      pandas.read_csv("data/ranked_domains/14-1-2021.csv", header=None)[1].tolist()][:100000]


WORDS = list(Counter(brown.words()).keys())
STOPWORDS = stopwords.words()
wordsegment.load()


########################################################################################################################
#                                          Preparation of text
########################################################################################################################

@indicate
def tokenize(text):
    return RegexpTokenizer(r'[^\W\d_]+').tokenize(text)  # without numbers

@indicate
def clear_text(word_raw):
    return [word for word in word_raw if word not in STOPWORDS and len(word) > 2]


########################################################################################################################
#                                          Text segmentation
########################################################################################################################


def segment(obj):
    return [word for str in [wordsegment.segment(word) for word in obj] for word in str]


########################################################################################################################
#                                          TF-IDF
########################################################################################################################

@indicate
def tokenize_url(word_raw):
    return segment(word_raw)


########################################################################################################################
#               URL hostname length
########################################################################################################################

@indicate
def url_length(url):
    return min(len(url) / 1169, 1)


########################################################################################################################
#               Count at ('@') symbol at base url
########################################################################################################################

@indicate
def count_at(base_url):
    return min(base_url.count('@') / 5, 1)


########################################################################################################################
#               Having semicolumn (;) symbol at base url
########################################################################################################################

@indicate
def count_semicolumn(url):
    return min(url.count(';') / 15, 1)


########################################################################################################################
#               Count equal (=) symbol at base url
########################################################################################################################

@indicate
def count_equal(base_url):
    return min(base_url.count('=') / 17, 1)


########################################################################################################################
#               Count percentage (%) symbol at base url
########################################################################################################################

@indicate
def count_percentage(base_url):
    return min(base_url.count('%') / 202, 1)


########################################################################################################################
#               Count dash (-) symbol at base url
########################################################################################################################

@indicate
def count_hyphens(base_url):
    return min(base_url.count('-') / 37, 1)


########################################################################################################################
#              Count number of dots in hostname
########################################################################################################################

@indicate
def count_dots(base_url):
    return min(base_url.count('.') / 35, 1)


########################################################################################################################
#              Count number of colon (:) symbol
########################################################################################################################

@indicate
def count_colon(url):
    return min(url.count(':') / 8, 1)


########################################################################################################################
#               Uses https protocol
########################################################################################################################

@indicate
def https_token(scheme):
    if scheme == 'https':
        return 0
    return 1


########################################################################################################################
#               Check if TLD in bad position
########################################################################################################################


def tld_in_path(tld, path):
    if path.lower().count(tld) > 0:
        return 1
    return 0


def tld_in_subdomain(tld, subdomain):
    if subdomain.count(tld) > 0:
        return 1
    return 0

@indicate
def tld_in_bad_position(tld, subdomain, path):
    if tld_in_path(tld, path) == 1 or tld_in_subdomain(tld, subdomain) == 1:
        return 1
    return 0


########################################################################################################################
#               Number of redirection
########################################################################################################################

@indicate
def count_redirection(page):
    return min(len(page.history) / 7, 1)


########################################################################################################################
#               Number of redirection to different domains
########################################################################################################################

@indicate
def count_external_redirection(page, domain):
    count = 0
    if len(page.history) == 0:
        return 0
    else:
        for i, response in enumerate(page.history, 1):
            if domain.lower() not in response.url.lower():
                count += 1
        return min(count / 4, 1)


########################################################################################################################
#               Is the registered domain created with random characters
########################################################################################################################

@indicate
def random_domain(second_level_domain):
    for word in segment([second_level_domain]):
        if word not in WORDS + brand_list:
            return 1

    return 0


###############################tld_in_path#########################################################################################
#               Presence of words with random characters
########################################################################################################################

@indicate
def random_words(words_raw, limit):
    return min(len([word for str in [segment([word]) for word in words_raw] for word in str if
                word not in WORDS + brand_list]) / limit, 1)


########################################################################################################################
#               domain in brand list
########################################################################################################################

@indicate
def domain_in_brand(second_level_domain):
    word = second_level_domain.lower()

    for idx, b in enumerate(brand_list):
        dst = len(Levenshtein.editops(word, b.lower()))
        if dst == 0:
            return 1 - idx / len(brand_list)
        elif dst <= (len(word) - 2) / 3 + 1:
            return 1 - idx / (len(brand_list) * 2)
    return 0


########################################################################################################################
#               count www in url words
########################################################################################################################

@indicate
def count_www(words_raw):
    count = 0
    for word in words_raw:
        if not word.find('www') == -1:
            count += 1
    return min(count / 5, 1)


########################################################################################################################
#               length of raw word list
########################################################################################################################

@indicate
def length_word_raw(words_raw):
    return min(len(words_raw) / 208, 1)


########################################################################################################################
#               count average word length in raw word list
########################################################################################################################

@indicate
def average_word_length(words_raw):
    if len(words_raw) == 0:
        return 0
    return min((sum(len(word) for word in words_raw) / len(words_raw) / 23), 1)


########################################################################################################################
#               longest word length in raw word list
########################################################################################################################

@indicate
def longest_word_length(words_raw):
    if len(words_raw) == 0:
        return 0
    return min(max(len(word) for word in words_raw) / 24, 1)


########################################################################################################################
#               Domain recognized by WHOIS
########################################################################################################################

@indicate
def whois_registered_domain(domain):
    try:
        hostname = whois.whois(domain).domain_name
        if type(hostname) == list:
            for host in hostname:
                if re.search(host.lower(), domain):
                    return 1
            return 0.5
        else:
            if re.search(hostname.lower(), domain):
                return 1
            else:
                return 0.5
    except:
        return 0


########################################################################################################################
#               Unable to get web traffic and page rank
########################################################################################################################


session = requests.session()

@indicate
def web_traffic(short_url):
    try:
        rank = BeautifulSoup(session.get("http://data.alexa.com/data?cli=10&dat=s&url=" + short_url, timeout=10).text,
                             "xml").find("REACH")['RANK']

        return min((int(rank) + 5e-07) / 10000000.0000005, 1)
    except:
        return 0


OPR_key = open("OPR_key.txt").read()

@indicate
def page_rank(domain):
    url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
    try:
        request = session.get(url, headers={'API-OPR': OPR_key}, timeout=7)
        result = request.json()
        result = result['response'][0]['page_rank_integer']
        if result:
            return (result + 1) / 11
        else:
            return 0
    except:
        return 0


########################################################################################################################
#               Certificate information
########################################################################################################################


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

@indicate
def get_cert(hostname):
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

    return result

@indicate
def count_alt_names(cert):
    try:
        return min((len(cert[b'subjectAltName'].split(',')) + 1) / 715, 1)
    except:
        return 0

@indicate
def valid_cert_period(cert):
    try:
        return min(((cert['notAfter'] - cert['notBefore']).days + 1)/1186, 1)
    except:
        return 0


########################################################################################################################
#               DNS record
########################################################################################################################

@indicate
def good_netloc(netloc):
    try:
        socket.gethostbyname(netloc)
        return 1
    except:
        return 0


########################################################################################################################
########################################################################################################################
#                                               HTML
########################################################################################################################
########################################################################################################################

@indicate
def urls_ratio(urls, total_urls):
    if len(total_urls) == 0:
        return 0
    else:
        return len(urls) / len(total_urls)


########################################################################################################################
#               ratio url-list
########################################################################################################################

@indicate
def ratio_List(Arr, key):
    total = len(Arr['internals']) + len(Arr['externals'])

    if 'embedded' in Arr:
        total += Arr['embedded']

    if total == 0:
        return 0
    elif key == 'embedded':
        return min(Arr[key] / total, 1)
    else:
        return min(len(Arr[key]) / total, 1)


########################################################################################################################
#               ratio of anchor
########################################################################################################################

@indicate
def ratio_anchor(Anchor, key):
    total = len(Anchor['safe']) + len(Anchor['unsafe'])

    if total == 0:
        return 0
    else:
        return len(Anchor[key]) / total


########################################################################################################################
########################################################################################################################
#                                               JAVASCRIPT
########################################################################################################################
########################################################################################################################

@indicate
def get_html_from_js(context):
    pattern = r"([\"'`])[\s\w]*(<\s*(\w+)[^>]*>.*(<\s*\/\s*\3\s*>)?)[\s\w]*\1"
    return " ".join([res.group(2) for res in re.finditer(pattern, context, re.MULTILINE) if res.group(2) is not None])

@indicate
def remove_JScomments(string):
    pattern = r"(\".*?\"|\'.*?\'|\`.*?\`)|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(2)

    return regex.sub(_replacer, string)


########################################################################################################################
#              Ratio static/dynamic html content
########################################################################################################################

@indicate
def ratio_dynamic_html(s_html, d_html):
    return max(0, min(1, len(d_html) / len(s_html)))


########################################################################################################################
#              Ratio html content on js-code
########################################################################################################################

@indicate
def ratio_js_on_html(html_context):
    if len(html_context):
        return len(get_html_from_js(remove_JScomments(html_context))) / len(html_context)
    else:
        return 0


########################################################################################################################
#              Amount of http request operations (popular)
########################################################################################################################

@indicate
def count_io_commands(string, limit):
    pattern = r"(\".*?\"|\'.*?\'|\`.*?\`)|" \
              r"((.(open|send)|$.(get|post|ajax|getJSON)|fetch|axios(|.(get|post|all))|getData)\s*\()"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    count = 0

    for m in re.finditer(regex, string):
        if not m.group(2) and m.groups():
            count += 1

    return min(count / limit, 1)


########################################################################################################################
#                       OCR
########################################################################################################################


def translate_image(obj):
    try:
        resp = requests.get(obj[0], stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        return pytesseract.image_to_string(img, lang=obj[1])
    except:
        return ""

@indicate
def image_to_text(img, lang):
    try:
        lang = languages.get(alpha2=lang).bibliographic

        if 'eng' not in lang:
            lang = 'eng+' + lang

        if type(img) == list:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                docs = [req for req in executor.map(translate_image, [(url, lang) for url in img], timeout=30)]

                if docs:
                    return ' '.join(docs)
                else:
                    return ""
        else:
            txt = pytesseract.image_to_string(img, lang=lang)
            return txt
    except:
        return ""


########################################################################################################################
#                   Relationship between image text and context
########################################################################################################################

@indicate
def ratio_Txt(dynamic, static):
    total = len(static)

    if total:
        return min(len(dynamic) / total, 1)
    else:
        return 0

@indicate
def count_phish_hints(word_raw, phish_hints, lang):
    if type(word_raw) == list:
        word_raw = ' '.join(word_raw).lower()

    try:
        exp = '|'.join(list(set([item for sublist in [phish_hints[lang], phish_hints['en']] for item in sublist])))

        if exp:
            return min(len(re.findall(exp, word_raw)) / 9, 1)
        else:
            return 0
    except:
        return 0


http_header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}


def is_URL_accessible(url, time_out=5):
    page = None

    if not url.startswith('http'):
        url = 'http://' + url

    try:
        page = requests.get(url, timeout=time_out, headers=http_header)
    except:
        parsed = urlparse(url)
        if not parsed.netloc.startswith('www'):
            url = parsed.scheme + '://www.' + parsed.netloc
            try:
                page = requests.get(url, timeout=time_out, headers=http_header)
            except:
                pass

    if page and page.status_code == 200 and page.content not in ["b''", "b' '"]:
        return True, page
    else:
        try:
            return False, page.status_code
        except:
            return False, -1

@indicate
def check_Language(text):
    global phish_hints

    language = translator.detect(str(text)[0:5000]).lang

    if language not in phish_hints.keys():
        words = translator.translate(" ".join(phish_hints['en'][:25]), src='en', dest=language).text.split(" ")

        phish_hints[language] = [str(word.strip()) for word in words]

        data = pandas.DataFrame(phish_hints[language])
        filename = "data/phish_hints/{0}.csv".format(language)
        data.to_csv(filename, index=False, header=False)

    return language

@indicate
def get_domain(url):
    o = urlsplit(url)
    return o.hostname, tldextract.extract(url).domain, o.path, o.netloc

@indicate
def extract_all_context_data(hostname, content, domain, base_url):
    Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                   "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

    Href = {'internals': [], 'externals': [], 'null': []}
    Link = {'internals': [], 'externals': [], 'null': []}
    Anchor = {'safe': [], 'unsafe': [], 'null': []}
    Img = {'internals': [], 'externals': [], 'null': []}
    Media = {'internals': [], 'externals': [], 'null': []}
    Form = {'internals': [], 'externals': [], 'null': []}
    SCRIPT = {'internals': [], 'externals': [], 'null': [], 'embedded': 0}  # JavaScript

    soup = BeautifulSoup(content, 'html.parser')

    # collect all external and internal hrefs from url
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

    # collect all external and internal hrefs from url
    for href in soup.find_all('a', href=True):
        url = href['href']

        if "#" in url or "javascript" in url.lower() or "mailto" in url.lower():
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

    # collect all media src tags
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

    # collect all link tags
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

    # collect all form actions
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

    # get content text
    Text = soup.get_text().lower()

    def merge_scripts(script_lnks):
        docs = []

        for url in script_lnks:
            state, request = is_URL_accessible(url)

            if state:
                docs.append(str(request.content))

        return "\n".join(docs)

    internals_script_doc = merge_scripts(SCRIPT['internals'])
    externals_script_doc = merge_scripts(SCRIPT['externals'])

    try:
        internals_script_doc = ' '.join(
            [internals_script_doc] + [script.contents[0] for script in soup.find_all('script', src=False) if
                                      len(script.contents) > 0])
        SCRIPT['embedded'] = len(
            [script.contents[0] for script in soup.find_all('script', src=False) if len(script.contents) > 0])
    except:
        pass

    return Href, Link, Anchor, Media, Img, Form, SCRIPT, Text, internals_script_doc, externals_script_doc

@indicate
def extract_text_context_data(content):
    return BeautifulSoup(content, 'html.parser').get_text().lower()


@indicate
def word_ratio(Text_words):
    if Text_words:
        return len(Counter(Text_words)) / len(Text_words)
    else:
        return 0

@indicate
def count_links(len):
    return min(len / 15585, 1)

@indicate
def count_words(len):
    return min(len / 990735, 1)


def extract_features(url):
    @indicate
    def words_raw_extraction(domain, subdomain, path):
        w_domain = re.split("[-./?=@&%:_]", domain.lower())
        w_subdomain = re.split("[-./?=@&%:_]", subdomain.lower())
        w_path = re.split("[-./?=@&%:_]", path.lower())
        raw_words = w_domain + w_path + w_subdomain
        w_host = w_domain + w_subdomain
        return segment(list(filter(None, raw_words))), \
               segment(list(filter(None, w_host))), \
               segment(list(filter(None, w_path)))

    (state, request) = is_URL_accessible(url)

    if state:
        r_url = request.url
        content = str(request.content)
        hostname, second_level_domain, path, netloc = get_domain(r_url)
        extracted_domain = tldextract.extract(r_url)
        domain = extracted_domain.domain + '.' + extracted_domain.suffix
        subdomain = extracted_domain.subdomain
        tmp = r_url[r_url.find(extracted_domain.suffix):len(r_url)]
        pth = tmp.partition("/")
        words_raw, words_raw_host, words_raw_path = words_raw_extraction(extracted_domain.domain, subdomain, pth[2])
        parsed = urlparse(r_url)
        scheme = parsed.scheme

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
            cert = e.submit(get_cert, domain).result()
            (Href, Link, Anchor, Media, Img, Form, SCRIPT, Text, internals_script_doc, externals_script_doc) = e.submit(
                extract_all_context_data, hostname, content, domain, r_url).result()

        content_di = get_html_from_js(remove_JScomments(internals_script_doc))
        content_de = get_html_from_js(remove_JScomments(externals_script_doc))

        Text_di = extract_text_context_data(content_di)
        Text_de = extract_text_context_data(content_de)

        lang = check_Language(content)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
            internals_img_txt = e.submit(image_to_text, Img['internals'], lang).result()
            externals_img_txt = e.submit(image_to_text, Img['externals'], lang).result()

        iImgTxt_words = clear_text(tokenize(internals_img_txt.lower()))
        eImgTxt_words = clear_text(tokenize(externals_img_txt.lower()))

        url_words = tokenize_url(words_raw)
        sContent_words = clear_text(tokenize(Text.lower()))
        diContent_words = clear_text(tokenize(Text_di.lower()))
        deContent_words = clear_text(tokenize(Text_de.lower()))

        Text_words = iImgTxt_words + eImgTxt_words + sContent_words + diContent_words + deContent_words

        iUrl_s = Href['internals'] + Link['internals'] + Media['internals'] + Form['internals']
        eUrl_s = Href['externals'] + Link['externals'] + Media['externals'] + Form['externals']
        nUrl_s = Href['null'] + Link['null'] + Media['null'] + Form['null']

        result = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as e:
            result.append(e.submit(word_ratio, Text_words).result())
            result.append(e.submit(good_netloc, netloc).result())
            result.append(e.submit(url_length, r_url).result())
            result.append(e.submit(count_at, r_url).result())
            result.append(e.submit(count_semicolumn, r_url).result())
            result.append(e.submit(count_equal, r_url).result())
            result.append(e.submit(count_percentage, r_url).result())
            result.append(e.submit(count_hyphens, r_url).result())
            result.append(e.submit(count_dots, r_url).result())
            result.append(e.submit(count_colon, r_url).result())
            result.append(e.submit(https_token, scheme).result())
            result.append(e.submit(count_phish_hints, url_words, phish_hints, lang).result())
            result.append(e.submit(count_redirection, request).result())
            result.append(e.submit(count_external_redirection, request, domain).result())
            result.append(e.submit(random_domain, second_level_domain).result())
            result.append(e.submit(random_words, words_raw, 86).result())
            result.append(e.submit(random_words, words_raw_host, 8).result())
            result.append(e.submit(domain_in_brand, second_level_domain).result())
            result.append(e.submit(count_www, words_raw).result())
            result.append(e.submit(length_word_raw, words_raw).result())
            result.append(e.submit(average_word_length, words_raw).result())
            result.append(e.submit(longest_word_length, words_raw).result())
            result.append(e.submit(count_links, len(iUrl_s) + len(eUrl_s)).result())
            result.append(e.submit(urls_ratio, iUrl_s, iUrl_s + eUrl_s + nUrl_s).result())
            result.append(e.submit(urls_ratio, nUrl_s, iUrl_s + eUrl_s + nUrl_s).result())
            result.append(e.submit(ratio_List, SCRIPT, 'internals').result())
            result.append(e.submit(ratio_List, Img, 'internals').result())
            result.append(e.submit(ratio_List, Media, 'externals').result())
            result.append(e.submit(ratio_anchor, Anchor, 'unsafe').result())
            result.append(e.submit(ratio_anchor, Anchor, 'safe').result())
            result.append(e.submit(count_words, len(sContent_words)).result())
            result.append(e.submit(ratio_Txt, iImgTxt_words + eImgTxt_words, sContent_words).result())
            result.append(e.submit(ratio_Txt, iImgTxt_words, sContent_words).result())
            result.append(e.submit(count_io_commands, internals_script_doc, 487490).result())
            result.append(e.submit(count_io_commands, externals_script_doc, 713513).result())
            result.append(e.submit(whois_registered_domain, domain).result())
            result.append(e.submit(web_traffic, r_url).result())
            result.append(e.submit(page_rank, domain).result())
            result.append(e.submit(valid_cert_period, cert).result())
            result.append(e.submit(count_alt_names, cert).result())

        return result
        # return [
        #     word_ratio(Text_words),
        #     good_netloc(netloc),
        #     url_length(r_url),
        #     count_at(r_url),
        #     count_semicolumn(r_url),
        #     count_equal(r_url),
        #     count_percentage(r_url),
        #     count_hyphens(r_url),
        #     count_dots(r_url),
        #     count_colon(r_url),
        #     https_token(scheme),
        #     count_phish_hints(url_words, phish_hints, lang),
        #     count_redirection(request),
        #     count_external_redirection(request, domain),
        #     random_domain(second_level_domain),
        #     random_words(words_raw, 86),
        #     random_words(words_raw_host, 8),
        #     domain_in_brand(second_level_domain),
        #     count_www(words_raw),
        #     length_word_raw(words_raw),
        #     average_word_length(words_raw),
        #     longest_word_length(words_raw),
        #     count_links(len(iUrl_s) + len(eUrl_s)),
        #     urls_ratio(iUrl_s, iUrl_s + eUrl_s + nUrl_s),
        #     urls_ratio(nUrl_s, iUrl_s + eUrl_s + nUrl_s),
        #     ratio_List(SCRIPT, 'internals'),
        #     ratio_List(Img, 'internals'),
        #     ratio_List(Media, 'externals'),
        #     ratio_anchor(Anchor, 'unsafe'),
        #     ratio_anchor(Anchor, 'safe'),
        #     count_words(len(sContent_words)),
        #     ratio_Txt(iImgTxt_words + eImgTxt_words, sContent_words),
        #     ratio_Txt(iImgTxt_words, sContent_words),
        #     count_io_commands(internals_script_doc, 487490),
        #     count_io_commands(externals_script_doc, 713513),
        #     whois_registered_domain(domain),
        #     web_traffic(r_url),
        #     page_rank(domain),
        #     valid_cert_period(cert),
        #     count_alt_names(cert)
        # ]
    return request


m = []

m.append(pickle.load(open('data/models/AdaBoost_DT/AdaBoost_DT.pkl', 'rb')))
m.append(pickle.load(open('data/models/Bagging_DT/Bagging_DT.pkl', 'rb')))
m.append(pickle.load(open('data/models/Bernoulli_NB/Bernoulli_NB.pkl', 'rb')))
m.append(pickle.load(open('data/models/Complement_NB/Complement_NB.pkl', 'rb')))
m.append(pickle.load(open('data/models/DT/DT.pkl', 'rb')))
m.append(pickle.load(open('data/models/ET/ET.pkl', 'rb')))
m.append(pickle.load(open('data/models/Gaussian_NB/Gaussian_NB.pkl', 'rb')))
m.append(pickle.load(open('data/models/GradientBoost/GradientBoost.pkl', 'rb')))
m.append(pickle.load(open('data/models/HistGradientBoost/HistGradientBoost.pkl', 'rb')))
m.append(pickle.load(open('data/models/KNN/kNN.pkl', 'rb')))
m.append(pickle.load(open('data/models/logistic_regression/logistic_regression.pkl', 'rb')))
m.append(pickle.load(open('data/models/Multinomial_NB/Multinomial_NB.pkl', 'rb')))
m.append(keras.models.load_model('data/models/neural_networks/ANN.h5'))
m.append(pickle.load(open('data/models/RF/RF.pkl', 'rb')))
m.append(pickle.load(open('data/models/Stacking (AB, RF, ET, B, HGB, GB, DT, XGB)/StackingClassifier.pkl', 'rb')))
m.append(pickle.load(open('data/models/Stacking (AB, RF, ET, B, XGB)/StackingClassifier.pkl', 'rb')))
m.append(pickle.load(open('data/models/Stacking (All)/StackingClassifier.pkl', 'rb')))
m.append(pickle.load(open('data/models/Stacking (ANN, LR, GB, HGB, XGB, AB)/StackingClassifier.pkl', 'rb')))
m.append(pickle.load(open('data/models/Stacking (GNB, BNB, CNB, MNB)/StackingClassifier.pkl', 'rb')))
m.append(pickle.load(open('data/models/Stacking (kNN, SVM, ANN)/StackingClassifier.pkl', 'rb')))
m.append(pickle.load(open('data/models/Stacking (kNN, SVM, ANN, DT, GNB, LR)/StackingClassifier.pkl', 'rb')))
m.append(pickle.load(open('data/models/Stacking (LR, GNB, BNB, CNB, MNB)/StackingClassifier.pkl', 'rb')))
m.append(pickle.load(open('data/models/SVM/SVM.pkl', 'rb')))
m.append(pickle.load(open('data/models/XGB/XGB.pkl', 'rb')))


estimators = [
    'extraction_data',
    "AdaBoost_DT",
    "Bagging_DT",
    "Bernoulli_NB",
    "Complement_NB",
    "DT",
    "ET",
    "Gaussian_NB",
    "GradientBoost",
    "HistGradientBoost",
    "kNN",
    "logistic_regression",
    "Multinomial_NB",
    "neural_networks",
    "RF",
    "Stacking (AB, RF, ET, B, HGB, GB, DT, XGB)",
    "Stacking (AB, RF, ET, B, XGB)",
    "Stacking (All)",
    "Stacking (ANN, LR, GB, HGB, XGB, AB)",
    "Stacking (GNB, BNB, CNB, MNB)",
    "Stacking (kNN, SVM, ANN)",
    "Stacking (kNN, SVM, ANN, DT, GNB, LR)",
    "Stacking (LR, GNB, BNB, CNB, MNB)",
    "SVM",
    "XGB"
]


from time import time


if __name__ == "__main__":
    @run_in_thread
    def check_site():
        result.configure(state='normal')
        result.delete(1.0, tk.END)

        global p_v, progress, timer
        p_v = 0
        progress['value'] = p_v

        dtime = []
        res = []
        timer = {}

        start = time()
        data = extract_features(url.get().split()[1])
        dtime.append(time() - start)

        if type(data) is list:
            data = np.array(extract_features(url.get().split()[1])).reshape((1, -1))

            result.configure(state='normal')
            result.insert(tk.END, " -> {} sec".format(dtime[-1]))

            for i in range(len(m)):
                start = time()

                r = m[i].predict_proba(data)
                dtime.append(time() - start)

                res.append(r.tolist()[0][-1])
                result.configure(state='normal')
                result.insert(tk.END, ('\n'+estimators[i+1], res[-1]))
                result.configure(state='disabled')
                p_v += 1
                result.configure(state='normal')
                result.insert(tk.END, " -> {} sec".format(dtime[-1]))
                result.configure(state='disabled')
                progress['value'] = p_v


            if os.path.isfile('data/logs/estimator_time.csv'):
                pandas.DataFrame(dtime).T.to_csv('data/logs/estimator_time.csv', header=False, mode='a', index=False)
            else:
                pandas.DataFrame(dtime).T.to_csv('data/logs/estimator_time.csv', header=estimators, index=False)

            df = pandas.read_csv('data/logs/estimator_time.csv')
            pandas.DataFrame([estimators, df.mean(), df.max(), df.min()]).T.to_csv('data/logs/estimator_avg_time.csv',
                                                                                   header=['estimator', 'mean', 'max',
                                                                                           'min'], index=False)

            df = pandas.DataFrame(timer.values()).T
            if os.path.isfile('data/logs/commands_timer.csv'):
                df.to_csv('data/logs/commands_timer.csv', header=False, mode='a', index=False)
            else:
                df.to_csv('data/logs/commands_timer.csv', header=list(timer.keys()), index=False)

            df = pandas.read_csv('data/logs/commands_timer.csv')
            pandas.DataFrame([list(timer.keys()), df.mean(), df.max(), df.min()]).T.to_csv(
                'data/logs/commands_timer_stats.csv',
                header=['estimator', 'mean', 'max',
                        'min'], index=False)

            phish = int(url.get().split()[0])

            if phish == 0:
                res = [1-r for r in res]

            df = pandas.DataFrame(res).T
            df[-1] = int(url.get().split()[0])

            if os.path.isfile('data/logs/estimator_rate.csv'):
                df.to_csv('data/logs/estimator_rate.csv', header=False, index=False, mode='a')
            else:
                df.to_csv('data/logs/estimator_rate.csv', header=estimators[1:]+['phish'], index=False)

            df = pandas.read_csv('data/logs/estimator_rate.csv')
            df.sum().to_csv('data/logs/estimator_rate_voite.csv', header=['count'], index_label='estimator')
            df.mean().to_csv('data/logs/estimator_rate_mean.csv', header=['rate'], index_label='estimator')

            df[df['phish'] == 0].sum().to_csv('data/logs/estimator_rate_legit.csv', header=['count'],
                                              index_label='estimator')
            df[df['phish'] == 1].sum().to_csv('data/logs/estimator_rate_phish.csv', header=['count'],
                                              index_label='estimator')

            df[df['phish'] == 0].mean().to_csv('data/logs/estimator_rate_mean_legit.csv', header=['rate'],
                                              index_label='estimator')
            df[df['phish'] == 1].mean().to_csv('data/logs/estimator_rate_mean_phish.csv', header=['rate'],
                                              index_label='estimator')
        else:
            result.configure(state='normal')
            result.insert(tk.END, "ERROR: {}".format(data))
            result.configure(state='disabled')

    window = tk.Tk()
    window.title("phishDetect")
    window.resizable(0, 0)

    mb = tk.Menubutton(window, text="settings", relief=tk.RAISED)
    mb.grid(column=0, row=0, sticky=tk.N+tk.W, pady=(0, 10))
    mb.menu = tk.Menu(mb, tearoff=0)
    mb["menu"] = mb.menu

    mayoVar = tk.IntVar()
    ketchVar = tk.IntVar()

    mb.menu.add_checkbutton(label="mayo",
                            variable=mayoVar)
    mb.menu.add_checkbutton(label="ketchup",
                            variable=ketchVar)


    def take_user_input_for_something():
        user_input = simpledialog.askstring("Pop up for user input!", "What do you want to ask the user to input here?")
        if user_input != "":
            print(user_input)


    mb.menu.add_command(label="Do something", command=take_user_input_for_something)

    url = tk.StringVar()

    textArea = tk.Entry(textvariable=url, width=80, exportselection=0)
    textArea.grid(column=0, row=1, sticky=tk.N+tk.S+tk.W+tk.E)

    btn = tk.Button(window, text="check", command=check_site)
    btn.grid(column=1, row=1, sticky=tk.N+tk.S+tk.W+tk.E)

    s = Style()
    s.theme_use("default")
    s.configure("TProgressbar", thickness=2)

    progress = Progressbar(
        window,
        orient=tk.HORIZONTAL,
        maximum=83,
        length=100,
        mode='determinate',
        style="TProgressbar"
    )
    progress.grid(column=0, row=2, columnspan=2,  sticky=tk.N + tk.S + tk.W + tk.E)

    scroll = tk.Scrollbar(window)
    scroll.grid(column=3, row=3, sticky=tk.N + tk.S + tk.W + tk.E)

    result = tk.Text(
        window,
        height=15,
        width=80,
        state='disabled',
        yscrollcommand=scroll.set
    )
    result.grid(column=0, row=3, columnspan=2)

    scroll.config(command=result.yview)

    window.mainloop()
