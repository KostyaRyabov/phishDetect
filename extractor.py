import sys
from colour import Color
from tldextract import extract as tld_extract
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from urllib.parse import urlparse, urlsplit, urljoin
from numpy import array, asarray
from googletrans import Translator
from cv2 import INTER_AREA, GaussianBlur, resize, IMREAD_COLOR, imdecode
from pytesseract import pytesseract, image_to_string
from bs4 import BeautifulSoup
from re import compile, finditer, MULTILINE, DOTALL, search, findall
import ssl
import socket
from OpenSSL.crypto import load_certificate, FILETYPE_PEM
from datetime import datetime
from whois import whois
from collections import Counter
from requests import session
from iso639 import languages
from threading import Thread
from pickle import load
from datetime import time
from wordninja import LanguageModel
import requests
from tkinter import END, Tk, StringVar, Entry, Button, N, S, W, E, Text, HORIZONTAL
from tkinter.ttk import Progressbar, Style
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


def indicate(func):
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return res
    return wrapper

def is_URL_accessible(url, time_out=3):
    page = None

    if not url.startswith('http'):
        url = 'http://' + url

    if urlparse(url).netloc.startswith('www.'):
        url = url.replace("www.", "", 1)

    try:
        page = requests.get(url, timeout=time_out, headers=http_header)
    except:
        pass

    if page:
        if page.status_code == 200 and page.content not in ["b''", "b' '"]:
            return True, page
        else:
            return 'HTTP Status Code: {}'.format(page.status_code)
    else:
        return False, 'Invalid Input!'

@indicate
def segment(text):
    return word_splitter.split(text)

@indicate
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

@indicate
def random_words(url_words):
    return sum([random_word(word) for word in url_words])

@indicate
def sld_in_brand(sld):
    if sld in brand_filter:
        return 1
    return 0

@indicate
def average_word_length(url_words):
    if len(url_words) == 0:
        return 0
    return sum(len(word) for word in url_words) / len(url_words)

@indicate
def longest_word_length(url_words):
    if len(url_words) == 0:
        return 0
    return max([len(word) for word in url_words])

session = session()

@indicate
def web_traffic(short_url):
    try:
        rank = BeautifulSoup(session.get("http://data.alexa.com/data?cli=10&dat=s&url=" + short_url, timeout=3).text,
                         "xml").find("REACH")['RANK']

        return min(int(rank) / 10000000, 1)
    except:
        return 1

@indicate
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

@indicate
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

@indicate
def valid_cert_period(cert):
    try:
        return (cert['notAfter'] - cert['notBefore']).days
    except:
        return 0

@indicate
def good_netloc(netloc):
    try:
        socket.gethostbyname(netloc)
        return 1
    except:
        return 0

@indicate
def urls_ratio(urls, total_urls):
    if len(total_urls) == 0:
        return 0
    else:
        return len(urls) / len(total_urls)

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

@indicate
def ratio_anchor(Anchor, key):
    total = len(Anchor['safe']) + len(Anchor['unsafe'])

    if total == 0:
        return 0
    else:
        return len(Anchor[key]) / total

@indicate
def get_html_from_js(context):
    pattern = r"([\"'`])[\s\w]*(<\s*(\w+)[^>]*>.*(<\s*\/\s*\3\s*>)?)[\s\w]*\1"
    return " ".join([r.group(2) for r in finditer(pattern, context, MULTILINE) if r.group(2) is not None])

@indicate
def remove_JScomments(string):
    pattern = r"(\".*?\"|\'.*?\'|\`.*?\`)|(/\*.*?\*/|//[^\r\n]*$)"
    regex = compile(pattern, MULTILINE | DOTALL)

    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(2)

    return regex.sub(_replacer, string)

@indicate
def ratio_js_on_html(html_context):
    if html_context:
        return len(get_html_from_js(remove_JScomments(html_context))) / len(html_context)
    else:
        return 0

@indicate
def count_io_commands(string):
    pattern = r"(\".*?\"|\'.*?\'|\`.*?\`)|" \
              r"((.(open|send)|$.(get|post|ajax|getJSON)|fetch|axios(|.(get|post|all))|getData)\s*\()"
    regex = compile(pattern, MULTILINE | DOTALL)

    count = 0

    for m in finditer(regex, string):
        if not m.group(2) and m.groups():
            count += 1

    return count

@indicate
def translate_image(obj):
    try:
        resp = session.get(obj[0], stream=True, timeout=2).raw
        image = asarray(bytearray(resp.read()), dtype="uint8")
        img = imdecode(image, IMREAD_COLOR)
        img = resize(img, None, fx=0.5, fy=0.5, interpolation=INTER_AREA)
        img = GaussianBlur(img, (5, 5), 0)

        return image_to_string(img, lang=obj[1])
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
            with ThreadPoolExecutor(25) as executor:
                for r in executor.map(translate_image, [(url, lang) for url in img], timeout=4):
                    if r:
                        docs.append(r)
            return "\n".join(docs)
        except TimeoutError:
            return "\n".join(docs)
    except:
        return ""

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
        word_raw = ' '.join(word_raw)

    try:
        exp = '|'.join(list(set([item for sublist in [phish_hints[lang], phish_hints['en']] for item in sublist])))

        if exp:
            return len(findall(exp, word_raw))
        else:
            return 0
    except:
        return 0

@indicate
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

@indicate
def get_domain(url):
    o = urlsplit(url)
    return o.hostname, tld_extract(url).domain, o.path, o.netloc

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
    SCRIPT = []

    CSS = {'internals': [], 'externals': [], 'null': [], 'embedded': 0}
    Favicon = {'internals': [], 'externals': [], 'null': []}

    soup = BeautifulSoup(content, 'lxml')

    @indicate
    def collector1():
        for script in soup.find_all('script', src=True):
            url = script['src']

            if url in Null_format:
                url = 'http://' + hostname + '/' + url
                Link['null'].append(url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Link['internals'].append(url)
            else:
                SCRIPT.append(url)
                Link['externals'].append(url)
    @indicate
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
    @indicate
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
    @indicate
    def merge_scripts(script_lnks):
        docs = []

        def load_script(url):
            state, request = is_URL_accessible(url, 1)

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

    with ThreadPoolExecutor(12) as e:
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
        Text = e.submit(soup.get_text).result().lower()
        externals_script_doc = e.submit(merge_scripts, SCRIPT).result()

    return Href, Link, Anchor, Media, Img, Form, CSS, Favicon, Text, externals_script_doc

@indicate
def word_ratio(Text_words):
    if Text_words:
        return len(Counter(Text_words)) / len(Text_words)
    else:
        return 0

@indicate
def having_ip_address(url):
    match = search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)|'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '[0-9a-fA-F]{7}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0

@indicate
def shortening_service(url):
    match = search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tr\.im|is\.gd|cli\.gs|yfrog\.com|'
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

@indicate
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

@indicate
def brand_in_path(words_raw_path):
    for word in words_raw_path:
        if word in brand_filter:
            return 1
    return 0

@indicate
def shortest_word_length(words_raw):
    if len(words_raw) == 0:
        return 0
    return min(len(word) for word in words_raw)

@indicate
def count_subdomain(netloc):
    return len(findall("\.", netloc))

@indicate
def compression_ratio(request):
    try:
        compressed_length = int(request.headers['content-length'])
        decompressed_length = len(request.content)
        return compressed_length / decompressed_length
    except:
        return 1

@indicate
def domain_expiration(whois_domain):
    try:
        expiration_date = whois_domain.expiration_date
        today = time.strftime('%Y-%m-%d')
        today = datetime.strptime(today, '%Y-%m-%d')

        if expiration_date:
            if type(expiration_date) == list:
                expiration_date = min(expiration_date)
            return abs((expiration_date - today).days)
        else:
            return 0
    except:
        return 0

@indicate
def extract_features(url):
    try:
        (state, request) = is_URL_accessible(url, 3)

        if state:
            request.encoding = 'utf-8'
            r_url = request.url
            content = request.text.lower()
            hostname, second_level_domain, path, netloc = get_domain(r_url)
            extracted_domain = tld_extract(r_url)
            domain = extracted_domain.domain + '.' + extracted_domain.suffix
            subdomain = extracted_domain.subdomain
            tmp = r_url[r_url.find(extracted_domain.suffix):len(r_url)]
            pth = tmp.partition("/")
            words_raw_path = segment(pth[2])
            cutted_url = extracted_domain.domain + subdomain
            words_raw_host = segment(cutted_url)
            cutted_url += pth[2]
            url_words = segment(cutted_url)

            with ThreadPoolExecutor(2) as e:
                cert = e.submit(get_cert, domain).result()
                (Href, Link, Anchor, Media, Img, Form, CSS, Favicon, Text, externals_script_doc) = e.submit(
                    extract_all_context_data, hostname, content, domain, r_url).result()

            with ThreadPoolExecutor(3) as e:
                lang = e.submit(check_Language, Text)
                iImgTxt_words = e.submit(image_to_text, Img['internals'], lang).result().lower()
                eImgTxt_words = e.submit(image_to_text, Img['externals'], lang).result().lower()

            with ThreadPoolExecutor(3) as e:
                iImgTxt_words = e.submit(reg.findall, iImgTxt_words).result()
                eImgTxt_words = e.submit(reg.findall, eImgTxt_words).result()
                sContent_words = e.submit(reg.findall, Text).result()

            Text_words = iImgTxt_words + eImgTxt_words + sContent_words

            iUrl_s = Href['internals'] + Link['internals'] + Media['internals'] + Form['internals']
            eUrl_s = Href['externals'] + Link['externals'] + Media['externals'] + Form['externals']
            nUrl_s = Href['null'] + Link['null'] + Media['null'] + Form['null']

            whois_domain = whois(domain)

            result = []
            with ThreadPoolExecutor(50) as e:
                result.append(e.submit(word_ratio, Text_words).result()),
                result.append(e.submit(having_ip_address, url).result()),
                result.append(e.submit(good_netloc, netloc).result()),
                result.append(e.submit(len, r_url).result()),
                result.append(e.submit(r_url.count, '@').result()),
                result.append(e.submit(r_url.count, ';').result()),
                result.append(e.submit(r_url.count, '&').result()),
                result.append(e.submit(r_url.count, '/', 2).result()),
                result.append(e.submit(r_url.count, '%').result()),
                result.append(e.submit(r_url.count, '-').result()),
                result.append(e.submit(r_url.count, '.').result()),
                result.append(e.submit(count_phish_hints, url_words, phish_hints, lang).result()),
                result.append(e.submit(len, url_words).result()),
                result.append(e.submit(len, request.history).result()),
                result.append(e.submit(count_external_redirection, request, domain).result()),
                result.append(e.submit(random_words, url_words).result()),
                result.append(e.submit(random_words, words_raw_host).result()),
                result.append(e.submit(char_repeat, url_words).result()),
                result.append(e.submit(char_repeat, words_raw_host).result()),
                result.append(e.submit(sld_in_brand, second_level_domain).result()),
                result.append(e.submit(brand_in_path, words_raw_path).result()),
                result.append(e.submit(cutted_url.count, 'www').result()),
                result.append(e.submit(cutted_url.count, 'com').result()),
                result.append(e.submit(average_word_length, url_words).result()),
                result.append(e.submit(longest_word_length, url_words).result()),
                result.append(e.submit(shortest_word_length, url_words).result()),
                result.append(e.submit(count_subdomain, netloc).result()),
                result.append(e.submit(compression_ratio, request).result()),
                result.append(e.submit(len, iUrl_s + eUrl_s).result()),
                result.append(e.submit(urls_ratio, iUrl_s, iUrl_s + eUrl_s + nUrl_s).result()),
                result.append(e.submit(urls_ratio, eUrl_s, iUrl_s + eUrl_s + nUrl_s).result()),
                result.append(e.submit(urls_ratio, nUrl_s, iUrl_s + eUrl_s + nUrl_s).result()),
                result.append(e.submit(ratio_List, CSS, 'internals').result()),
                result.append(e.submit(ratio_List, Img, 'externals').result()),
                result.append(e.submit(ratio_List, Img, 'internals').result()),
                result.append(e.submit(ratio_List, Favicon, 'internals').result()),
                result.append(e.submit(ratio_List, Media, 'internals').result()),
                result.append(e.submit(ratio_List, Media, 'externals').result()),
                result.append(e.submit(ratio_anchor, Anchor, 'unsafe').result()),
                result.append(e.submit(ratio_anchor, Anchor, 'safe').result()),
                result.append(e.submit(ratio_List, Link, 'externals').result()),
                result.append(e.submit(count_phish_hints, Text, phish_hints, lang).result()),
                result.append(e.submit(len, sContent_words).result()),
                result.append(e.submit(ratio_Txt, iImgTxt_words + eImgTxt_words, sContent_words).result()),
                result.append(e.submit(ratio_Txt, iImgTxt_words, sContent_words).result()),
                result.append(e.submit(count_io_commands, externals_script_doc).result()),
                result.append(e.submit(domain_expiration, whois_domain).result()),
                result.append(e.submit(web_traffic, r_url).result()),
                result.append(e.submit(page_rank, domain).result()),
                result.append(e.submit(valid_cert_period, cert).result())

            return result
        return request
    except Exception as ex:
        return ex

d = [
    1.0, 1.0, 1.0, 1384.0, 4.0, 12.0, 17.0, 21.0, 152.0, 30.0, 27.0, 30.0, 1071.0, 8.0, 6.0, 640.0, 17.0, 0.5555555555555555, 0.6666666666666666, 1.0, 1.0, 3.0, 6.0, 23.0, 36.0, 23.0, 11.0, 4.28868961950903, 16394.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3978.0, 998427.0, 1.0, 1.0, 136276.0, 31908.0, 0.9999999, 10.0, 825.0
]

from tqdm import tqdm
from concurrent.futures import as_completed
import pandas as pd

from threading import Lock

rec_locker = Lock()

def generate_dataset(url_list):
    def extraction_data(obj):
        try:
            data = extract_features(obj[0])

            if type(data) is list:
                res = array([max(min(data[i] / d[i], 1), 0) for i in range(50)]) * 0.998 + 0.001

                with rec_locker:
                    pd.DataFrame(res.tolist() + [obj[1]]).T.to_csv('data/datasets/OUTPUT/append.csv', mode='a', index=False, header=False)
        except Exception as ex:
            print(ex)

    with ThreadPoolExecutor(max_workers=15) as executor:
        fut = [executor.submit(extraction_data, url) for url in url_list]
        for _ in tqdm(as_completed(fut), total=len(url_list)):
            pass