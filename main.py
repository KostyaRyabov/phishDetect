import sys
from colour import Color
import tldextract
import concurrent.futures
from urllib.parse import urlparse, urlsplit, urljoin
from nltk.tokenize import RegexpTokenizer
import wordsegment
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

import tkinter as tk
from tkinter.ttk import Progressbar, Style

p_v = 0
progress = {'value': 0}


def indicate(func):
    def wrapper(*args, **kwargs):
        global progress, p_v
        res = func(*args, **kwargs)
        p_v += 1
        progress['value'] = p_v
        return res

    return wrapper


progress_task = None


class KThread(threading.Thread):
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
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


http_header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

translator = Translator()
wordsegment.load()

phish_hints = pickle.load(open('phish_hints.pkl', 'rb'))
brand_list = pickle.load(open('brand_list.pkl', 'rb'))
WORDS = pickle.load(open('words.pkl', 'rb'))
STOPWORDS = pickle.load(open('stopwords.pkl', 'rb'))
classifier = pickle.load(open('classifier.pkl', 'rb'))


########################################################################################################################
#                                          Preparation of text
########################################################################################################################

@indicate
def tokenize(text):
    return RegexpTokenizer(r'[^\W\d_]+').tokenize(text)  # without numbers


def cf(word):
    if word not in STOPWORDS and len(word) > 2:
        return word
    return None

@indicate
def clear_text(word_raw):
    docs = []

    try:
        with concurrent.futures.ThreadPoolExecutor(25) as executor:
            res = executor.map(translate_image, [(word, cf) for word in word_raw], timeout=3)

            for r in res:
                if r:
                    docs.append(r)

        return docs
    except concurrent.futures._base.TimeoutError:
        return docs


########################################################################################################################
#                                          Text segmentation
########################################################################################################################


def segment(obj):
    return [word for str in [wordsegment.segment(word) for word in obj] for word in str]


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


def random_domain(second_level_domain):
    for word in segment([second_level_domain]):
        if word not in WORDS + brand_list:
            return 1

    return 0


###############################tld_in_path#########################################################################################
#               Presence of words with random characters
########################################################################################################################

@indicate
def random_words(url_words, limit):
    return min(len([word for str in [segment([word]) for word in url_words] for word in str if
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
def count_www(url_words):
    count = 0
    for word in url_words:
        if not word.find('www') == -1:
            count += 1
    return min(count / 5, 1)


########################################################################################################################
#               length of raw word list
########################################################################################################################

@indicate
def length_word_raw(url_words):
    return min(len(url_words) / 208, 1)


########################################################################################################################
#               count average word length in raw word list
########################################################################################################################

@indicate
def average_word_length(url_words):
    if len(url_words) == 0:
        return 0
    return min((sum(len(word) for word in url_words) / len(url_words) / 23), 1)


########################################################################################################################
#               longest word length in raw word list
########################################################################################################################

@indicate
def longest_word_length(url_words):
    if len(url_words) == 0:
        return 0
    return min(max(len(word) for word in url_words) / 24, 1)


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
        rank = BeautifulSoup(session.get("http://data.alexa.com/data?cli=10&dat=s&url=" + short_url, timeout=3).text,
                             "xml").find("REACH")['RANK']

        return min((int(rank) + 5e-07) / 10000000.0000005, 1)
    except:
        return 1


OPR_key = open("OPR_key.txt").read()

@indicate
def page_rank(domain):
    url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
    try:
        request = session.get(url, headers={'API-OPR': OPR_key}, timeout=3)
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


def get_certificate(host, port=443, timeout=3):
    context = ssl.create_default_context()
    conn = socket.create_connection((host, port), timeout=timeout)
    sock = context.wrap_socket(conn, server_hostname=host)
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
        resp = requests.get(obj[0], stream=True, timeout=3).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        return pytesseract.image_to_string(img, lang=obj[1])
    except:
        return ""


@indicate
def image_to_text(img, lang):
    if not img:
        return ""

    try:
        lang = languages.get(alpha2=lang).bibliographic

        if 'eng' not in lang:
            lang = 'eng+' + lang

        docs = []

        try:
            with concurrent.futures.ThreadPoolExecutor(25) as executor:
                res = executor.map(translate_image, [(url, lang) for url in img], timeout=15)

                for r in res:
                    if r:
                        docs.append(r)

            return "\n".join(docs)
        except concurrent.futures._base.TimeoutError:
            return "\n".join(docs)
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


def is_URL_accessible(url, time_out=3):
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

    size = len(text)
    if size > 10000:
        size = 10000

    language = translator.detect(str(text)[:size]).lang

    if type(language) is list:
        if 'en' in language:
            language.remove('en')
        language = language[-1]

    return language

@indicate
def get_domain(url):
    o = urlsplit(url)
    return o.hostname, tldextract.extract(url).domain, o.path, o.netloc

@indicate
def extract_all_context_data(hostname, content, domain, base_url):
    global p_v

    Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                   "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

    Href = {'internals': [], 'externals': [], 'null': []}
    Link = {'internals': [], 'externals': [], 'null': []}
    Anchor = {'safe': [], 'unsafe': [], 'null': []}
    Img = {'internals': [], 'externals': [], 'null': []}
    Media = {'internals': [], 'externals': [], 'null': []}
    Form = {'internals': [], 'externals': [], 'null': []}
    SCRIPT = {'internals': [], 'externals': [], 'null': [], 'embedded': 0}  # JavaScript

    soup = BeautifulSoup(content, 'lxml')

    @indicate
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


    @indicate
    def collector2():
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

    @indicate
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

    @indicate
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

    @indicate
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

    @indicate
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

    @indicate
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

    @indicate
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

    @indicate
    def merge_scripts(script_lnks):
        docs = []

        def load_script(url):
            state, request = is_URL_accessible(url)

            if state:
                docs.append(str(request.content))

        try:
            with concurrent.futures.ThreadPoolExecutor(25) as executor:
                res = executor.map(load_script, script_lnks, timeout=15)

                for r in res:
                    if r:
                        docs.append(r)
            return "\n".join(docs)
        except concurrent.futures._base.TimeoutError:
            return "\n".join(docs)

    with concurrent.futures.ThreadPoolExecutor(11) as e:
        e.submit(collector1)
        e.submit(collector2)
        e.submit(collector3)
        e.submit(collector4)
        e.submit(collector5)
        e.submit(collector6)
        e.submit(collector7)
        e.submit(collector8)
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

    return Href, Link, Anchor, Media, Img, Form, SCRIPT, Text, internals_script_doc, externals_script_doc

@indicate
def extract_text_context_data(content):
    return BeautifulSoup(content, 'lxml').get_text().lower()


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
    try:
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
            url_words, words_raw_host, words_raw_path = words_raw_extraction(extracted_domain.domain, subdomain, pth[2])
            parsed = urlparse(r_url)
            scheme = parsed.scheme

            with concurrent.futures.ThreadPoolExecutor(2) as e:
                cert = e.submit(get_cert, domain).result()
                (Href, Link, Anchor, Media, Img, Form, SCRIPT, Text, internals_script_doc, externals_script_doc) = e.submit(
                    extract_all_context_data, hostname, content, domain, r_url).result()

            content_di = get_html_from_js(remove_JScomments(internals_script_doc))
            content_de = get_html_from_js(remove_JScomments(externals_script_doc))

            Text_di = extract_text_context_data(content_di)
            Text_de = extract_text_context_data(content_de)

            lang = check_Language(content)

            with concurrent.futures.ThreadPoolExecutor(2) as e:
                internals_img_txt = e.submit(image_to_text, Img['internals'], lang).result()
                externals_img_txt = e.submit(image_to_text, Img['externals'], lang).result()

            iImgTxt_words = clear_text(tokenize(internals_img_txt.lower()))
            eImgTxt_words = clear_text(tokenize(externals_img_txt.lower()))

            sContent_words = clear_text(tokenize(Text.lower()))
            diContent_words = clear_text(tokenize(Text_di.lower()))
            deContent_words = clear_text(tokenize(Text_de.lower()))

            Text_words = iImgTxt_words + eImgTxt_words + sContent_words + diContent_words + deContent_words

            iUrl_s = Href['internals'] + Link['internals'] + Media['internals'] + Form['internals']
            eUrl_s = Href['externals'] + Link['externals'] + Media['externals'] + Form['externals']
            nUrl_s = Href['null'] + Link['null'] + Media['null'] + Form['null']

            result = []

            with concurrent.futures.ThreadPoolExecutor(35) as e:
                result.append(e.submit(word_ratio, Text_words).result())
                result.append(e.submit(url_length, r_url).result())
                result.append(e.submit(count_semicolumn, r_url).result())
                result.append(e.submit(count_hyphens, r_url).result())
                result.append(e.submit(count_dots, r_url).result())
                result.append(e.submit(https_token, scheme).result())
                result.append(e.submit(count_phish_hints, url_words, phish_hints, lang).result())
                result.append(e.submit(count_redirection, request).result())
                result.append(e.submit(count_external_redirection, request, domain).result())
                result.append(e.submit(random_domain, second_level_domain).result())
                result.append(e.submit(random_words, url_words, 86).result())
                result.append(e.submit(random_words, words_raw_host, 8).result())
                result.append(e.submit(domain_in_brand, second_level_domain).result())
                result.append(e.submit(count_www, url_words).result())
                result.append(e.submit(length_word_raw, url_words).result())
                result.append(e.submit(average_word_length, url_words).result())
                result.append(e.submit(longest_word_length, url_words).result())
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
        return 'HTTP Status Code: '.format(request)
    except Exception as ex:
        return ex


if __name__ == "__main__":
    @run_in_thread
    def check_site():
        result.configure(background='white')
        result.configure(state='normal')
        result.delete(1.0, tk.END)
        result.configure(state='disabled')

        global p_v, progress
        p_v = 1
        progress['value'] = p_v

        data = extract_features(url.get().strip())

        if type(data) is list:
            data = np.array(data).reshape((1, -1)) * 0.998 + 0.001

            res = classifier.predict_proba(data).tolist()[0][-1]

            result.configure(state='normal')
            result.configure(background=Color(hsl=(0.2778*(1-res), 1, 0.5)).get_hex_l())

            if res < 0.5:
                result.insert(tk.END, "\nЭто легитимный сайт!".format((1-res)*100), 'tag-center')
            else:
                result.insert(tk.END, "\nЭто фишинговый сайт!".format(res * 100), 'tag-center')

            result.configure(state='disabled')
        else:
            result.configure(state='normal')
            result.insert(tk.END, "ERROR: {}".format(data))
            result.configure(state='disabled')
        progress['value'] = 69

    window = tk.Tk()
    window.title("phishDetect")
    window.resizable(0, 0)

    url = tk.StringVar()

    textArea = tk.Entry(textvariable=url, width=80, exportselection=0)
    textArea.grid(column=0, row=0, sticky=tk.N+tk.S+tk.W+tk.E)

    btn = tk.Button(window, text="check", command=check_site)
    btn.grid(column=1, row=0, sticky=tk.N+tk.S+tk.W+tk.E)

    s = Style()
    s.theme_use("default")
    s.configure("TProgressbar", thickness=2)

    progress = Progressbar(
        window,
        orient=tk.HORIZONTAL,
        maximum=69,
        length=100,
        mode='determinate',
        style="TProgressbar"
    )
    progress.grid(column=0, row=1, columnspan=2,  sticky=tk.N + tk.S + tk.W + tk.E)

    result = tk.Text(
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