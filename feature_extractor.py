from tqdm import tqdm
import re
import requests
import pandas
import concurrent.futures
import lxml
import time
from threading import Lock
from random import randint
from datetime import date
import numpy as np
from sklearn import svm
import pickle

from sklearn.model_selection import train_test_split
from tldextract import extract as tld_extract
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from urllib.parse import urlparse, urlsplit, urljoin
from numpy import asarray
from googletrans import Translator
from cv2 import INTER_AREA, GaussianBlur, resize, IMREAD_COLOR, imdecode
from pytesseract import pytesseract, image_to_string
from bs4 import BeautifulSoup
from re import compile, finditer, MULTILINE, DOTALL, split, search, findall
import ssl
import socket
from OpenSSL.crypto import load_certificate, FILETYPE_PEM
from datetime import datetime
import whois
from collections import Counter
from requests import session
from iso639 import languages
from pickle import load

from feature_selector import FeatureSelector

from bloomfpy import BloomFilter
import wordninja

word_splitter = wordninja.LanguageModel('wordlist.txt.gz')
brand_filter = pickle.load(open('Trie_brands.pkl', 'rb'))
words_filter = pickle.load(open('Trie_words_Pro.pkl', 'rb'))

key = open("OPR_key.txt").read()
translator = Translator()
phish_hints = pickle.load(open('phish_hints.pkl', 'rb'))

pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

reg = compile(r'\w{3,}')

classifier = load(open('classifier.pkl', 'rb'))

http_header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
    'Content-Type': "text/html; charset=utf-8"}

headers = [
        'коэффициент уникальности слов',
        'наличие ip-адреса в url',
        'сокращение url',
        "наличие сертификата",
        "хороший netloc",
        'длина url',
        'кол-во @ в url',
        'кол-во ! в url',
        'кол-во + в url',
        'кол-во [ и ] в url',
        'кол-во ( и ) в url',
        'кол-во , в url',
        'кол-во $ в url',
        'кол-во ; в url',
        'кол-во пробелов в url',
        'кол-во & в url',
        'кол-во // в url',
        'кол-во / в url',
        'кол-во = в url',
        'кол-во % в url',
        'кол-во ? в url',
        'кол-во _ в url',
        'кол-во - в url',
        'кол-во . в url',
        'кол-во : в url',
        'кол-во * в url',
        'кол-во | в url',
        'кол-во ~ в url',
        'кол-во http токенов в url',
        'https',
        'соотношение цифр в url',
        'кол-во цифр в url',
        "фишинговые слова в url",
        "кол-во слов в url",
        'tld в пути',
        'tld в поддомене',
        'tld на плохой позиции',
        'ненормальный поддомен',
        'внутренние перенаправления',
        'внешние перенаправления',
        'случайный домен',
        'случайные слова в url',
        'случайные слова в хосте url',
        'случайные слова в пути url',
        'повторяющиеся символы в url',
        'повторяющиеся символы в хосте url',
        'повторяющиеся символы в пути url',
        'наличие punycode',
        'домен в брендах',
        'бренд в пути url',
        'кол-во www в url',
        'кол-во com в url',
        'наличие порта в url',
        'кол-во слов в url2',
        'средняя длина слова в url',
        'максимальная длина слова в url',
        'минимальная длина слова в url',
        'префикс и суффикс в url',
        'кол-во поддоменов',
        'тайпсквотинг',
        'сжатие страницы',
        'ввод/вывод в основном контексте',
        'кол-во кода',
        'кол-во ссылок в контексте',
        'кол-во внутренних ссылок',
        'кол-во внешних ссылок',
        'кол-во пустых ссылок',
        "кол-во внутренних CSS",
        "кол-во внешних CSS",
        "кол-во встроенных CSS",
        "кол-во внутренних скриптов",
        "кол-во внешних скриптов",
        "кол-во встроенных скриптов",
        "кол-во внешних изображений",
        "кол-во внутренних изображений",
        "перенаправления по внутренним ссылкам",
        "перенаправления по внешним ссылкам",
        "ошибки во внутренних ссылках",
        "ошибки во внешних ссылках",
        "форма входа",
        "внешний Favicon",
        "внутренний Favicon",
        "письмо на почту",
        "внутренние медиа",
        "внешние медиа",
        "пустой титульник",
        "кол-во небезопасных якорей",
        "кол-во безопасных якорей",
        "кол-во внутренних ресурсов",
        "кол-во внешних ресурсов",
        "невидимый iframe",
        "onmouseover",
        "всплывающие окна",
        "события правой кнопки мыши",
        "домен в тексте",
        "домена в титульнике",
        "домен с авторскими правами",
        "фишинговые слова в тексте",
        "кол-во слов в тексте",
        "объем текста изображений",
        "объем текста внутренних изображений",
        "объем текста внешних изображений",
        "соотношение текста изображений",
        "объем динамического текста",
        "объем внутреннее добавляемого текста",
        "кол-во внутренне добавляемого кода",
        "внутренне добавляемый ввод/вывод",
        'кол-во внутренне добавляемых ссылок',
        'кол-во внутренне добавляемых внутренних ссылок',
        'кол-во внутренне добавляемых внешних ссылок',
        'кол-во внутренне добавляемых пустых ссылок',
        "кол-во внутренне добавляемых внутренних CSS",
        "кол-во внутренне добавляемых внешних CSS",
        "кол-во внутренне добавляемых встроенных CSS",
        "кол-во внутренне добавляемых внутренних скриптов",
        "кол-во внутренне добавляемых внешних скриптов",
        "кол-во внутренне добавляемых встроенных скриптов",
        "кол-во внутренне добавляемых внешних изображений",
        "кол-во внутренне добавляемых внутренних изображений",
        "перенаправления внутренне добавляемых внутренних ссылок",
        "перенаправления внутренне добавляемых внешних ссылок",
        "ошибки внутренне добавляемых внутренних ссылок",
        "ошибки внутренне добавляемых внешних ссылок",
        "внутренне добавляемая форма входа",
        "внутренне добавляемые внешние Favicon",
        "внутренне добавляемые внутренних Favicon",
        "внутренне добавляемые письма на почту",
        "внутренне добавляемые внешние медиа",
        "внутренне добавляемые внутренние медиа",
        "внутренне добавляемый пустой титульник",
        "внутренне добавляемые небезопасные якори",
        "внутренне добавляемые безопасные якори",
        "внутренне добавляемые внутренние ресурсы",
        "внутренне добавляемые внешние ресурсы",
        "внутренне добавляемые невидимые iframe",
        "внутренне добавляемые onmouseover",
        "внутренне добавляемые всплывающие окна",
        "внутренне добавляемые события правой кнопки мыши",
        "домен во внутренне добавляемом тексте",
        "домен во внутренне добавляемом титульнике",
        "домен с внутренне добавляемыми авторскими правами",
        "ввод/вывод во внутренне добавляемом коде",
        "фишинговые слова во внутренне добавляемом",
        "внутренне добавляемый объем текста",
        "объем внешне добавляемого текста",
        "кол-во внешне добавляемого кода",
        "внешне добавляемый ввод/вывод",
        'кол-во внешне добавляемых ссылок',
        'кол-во внешне добавляемых внутренних ссылок',
        'кол-во внешне добавляемых внешних ссылок',
        'кол-во внешне добавляемых пустых ссылок',
        "кол-во внешне добавляемых внутренних CSS",
        "кол-во внешне добавляемых внешних CSS",
        "кол-во внешне добавляемых встроенных CSS",
        "кол-во внешне добавляемых внутренних скриптов",
        "кол-во внешне добавляемых внешних скриптов",
        "кол-во внешне добавляемых встроенных скриптов",
        "кол-во внешне добавляемых внешних изображений",
        "кол-во внешне добавляемых внутренних изображений",
        "перенаправления внешне добавляемых внутренних ссылок",
        "перенаправления внешне добавляемых внешних ссылок",
        "ошибки внешне добавляемых внутренних ссылок",
        "ошибки внешне добавляемых внешних ссылок",
        "внешне добавляемая форма входа",
        "внешне добавляемые внешние Favicon",
        "внешне добавляемые внутренних Favicon",
        "внешне добавляемые письма на почту",
        "внешне добавляемые внешние медиа",
        "внешне добавляемые внутренние медиа",
        "внешне добавляемый пустой титульник",
        "внешне добавляемые небезопасные якори",
        "внешне добавляемые безопасные якори",
        "внешне добавляемые внутренние ресурсы",
        "внешне добавляемые внешние ресурсы",
        "внешне добавляемые невидимые iframe",
        "внешне добавляемые onmouseover",
        "внешне добавляемые всплывающие окна",
        "внешне добавляемые события правой кнопки мыши",
        "домен во внешне добавляемом тексте",
        "домен во внешне добавляемом титульнике",
        "домен с внешне добавляемыми авторскими правами",
        "ввод/вывод во внешне добавляемом коде",
        "фишинговые слова во внешне добавляемом",
        "внешне добавляемый объем текста",
        'срок регистрации домена',
        "домен зарегистрирован",
        "рейтинг по Alexa",
        "рейтинг по openpagerank",
        "времени действия сертификата",
        "срок действия сертификата",
        "кол-во альтернативных имен"
    ]


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

def extract_URLs_from_page(content, domain, base_url):
    Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                   "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

    Href = []
    soup = BeautifulSoup(content, 'lxml')

    for href in soup.find_all('a', href=True)[:20]:
        url = href['href']

        if url in Null_format:
            continue

        url = urljoin(base_url, url)

        if domain in urlparse(url).netloc:
            Href.append(url)

    return Href

def generate_legitimate_urls(N, seed=0):
    domain_list = pandas.read_csv("data/ranked_domains/14-1-2021.csv", header=None)[1].tolist()[seed:]

    url_list = []

    def url_search(domain):
        if len(url_list) >= N:
            return

        url = search_for_vulnerable_URLs(domain)

        if url:
            url_list.append(url)

            pandas.DataFrame([url]).to_csv(
                "data/urls/legitimate/{0}.csv".format(date.today().strftime("%d-%m-%Y")), index=False, header=False,
                mode='a')

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        fut = [executor.submit(url_search, domain) for domain in domain_list]
        for _ in tqdm(concurrent.futures.as_completed(fut), total=len(domain_list)):
            pass

def search_for_vulnerable_URLs(domain):
    state, request = is_URL_accessible(domain, 1)

    if state:
        url = request.url
        content = request.text

        Href = extract_URLs_from_page(content, domain, url)

        if Href:
            url = Href[randint(0, len(Href))-1]
            state, request = is_URL_accessible(url)
            if state:
                return request.url

    return None


from threading import Semaphore

req_img_locker = Semaphore(2)
req_links_locker = Semaphore(8)
record_lock = Lock()


def generate_dataset(url_list):
    def extraction_data(obj):
        try:
            res = extract_features(obj[0])

            if type(res) is list:
                with record_lock:
                    pandas.DataFrame(res + [obj[1]]).T.to_csv('data/datasets/RAW/1.csv', mode='a', index=False, header=False)
        except Exception as ex:
            print(ex)

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        fut = [executor.submit(extraction_data, url) for url in url_list]
        for _ in tqdm(concurrent.futures.as_completed(fut), total=len(url_list)):
            pass


from sklearn.model_selection import KFold


def RFE(X, Y, step=1):
    N = 0

    accuracies = {}
    bestX = None

    clfRFE = svm.SVC(kernel='linear', verbose=True, tol=0.001, max_iter=100000)

    featureCount = X.shape[1]
    featureList = np.arange(0, featureCount)
    included = np.full(featureCount, True)
    curCount = featureCount

    maxV = 0

    while curCount > N:
        actualFeatures = featureList[included]
        Xnew = X.iloc[:, actualFeatures]

        x_train, x_test, y_train, y_test = train_test_split(Xnew, Y, test_size=0.15, random_state=5)

        clfRFE.fit(x_train, y_train)
        accuracies[curCount] = clfRFE.score(x_test, y_test)

        print('\t[step: {}, acc: {}]'.format(curCount, accuracies[curCount]))

        if accuracies[curCount] >= maxV:
            maxV = accuracies[curCount]
            bestX = X.iloc[:, included]

        curStep = min(step, curCount - N)
        elim = np.argsort(np.abs(clfRFE.coef_[0]))[:curStep]
        included[actualFeatures[elim]] = False
        curCount -= curStep

    pandas.DataFrame().from_dict(accuracies, orient='index').to_csv('data/datasets/PROCESS/sizing.csv', header=False, index=False)

    # f = pd.read_csv('data/datasets/PROCESS/sizing.csv', header=None, index_col=None)
    # v = f[0].values.tolist() + [0.65]
    # v.reverse()
    # importance_index = np.argmax(v)
    # pyplot.plot(v, 'r')
    # pyplot.vlines(x=importance_index, ymin=np.min(v), ymax=np.max(v), linestyles='--',
    #               colors='blue')
    # pyplot.show()

    return bestX


def select_features():
    frame = pandas.read_csv('data/datasets/RAW/dataset.csv', header=None, index_col=None)

    end = len(frame.columns) - 1

    df0 = frame[frame[end] == 0]
    df1 = frame[frame[end] == 1]

    d = len(df0) - len(df1)

    if d > 0:
        frame = pandas.concat([df0.iloc[:-d], df1])
    elif d < 0:
        frame = pandas.concat([df0, df1.iloc[:d]])

    pandas.DataFrame([
        headers,
        frame[frame[end]==0].max().to_list()[:-1],
        frame[frame[end]==0].min().to_list()[:-1],
        frame[frame[end]==0].mean().to_list()[:-1]
    ]).T.to_csv("data/datasets/RAW/legit_stats.csv", index=False,
                header=['feature', 'max', 'min', 'mean'])

    pandas.DataFrame([
        headers,
        frame[frame[end] == 1].max().to_list()[:-1],
        frame[frame[end] == 1].min().to_list()[:-1],
        frame[frame[end] == 1].mean().to_list()[:-1]
    ]).T.to_csv("data/datasets/RAW/phish_stats.csv", index=False,
                header=['feature', 'max', 'min', 'mean'])

    # нормализация

    frame.columns = headers + ['status']

    X = frame[headers]
    Y = frame['status']

    a = 0.001
    b = 0.999

    X = a + (X - X.min()) / (X.max() - X.min()) * (b-a)

    tmp = X
    tmp['status'] = Y
    tmp.to_csv("data/datasets/PROCESS/dataset.csv", index=False)

    pandas.DataFrame([
        headers,
        tmp[tmp['status'] == 0].max().to_list()[:-1],
        tmp[tmp['status'] == 0].min().to_list()[:-1],
        tmp[tmp['status'] == 0].mean().to_list()[:-1]
    ]).T.to_csv("data/datasets/PROCESS/legit_stats.csv", index=False,
                header=['feature', 'max', 'min', 'mean'])

    pandas.DataFrame([
        headers,
        tmp[tmp['status'] == 1].max().to_list()[:-1],
        tmp[tmp['status'] == 1].min().to_list()[:-1],
        tmp[tmp['status'] == 1].mean().to_list()[:-1]
    ]).T.to_csv("data/datasets/PROCESS/phish_stats.csv", index=False,
                header=['feature', 'max', 'min', 'mean'])

    del tmp

    X = frame[headers]
    X = a + (X - X.min()) / (X.max() - X.min()) * (b - a)

    fs = FeatureSelector(data=X, labels=Y)

    fs.identify_all({
        'missing_threshold': 0.6,
        'correlation_threshold': 0.98,
        'eval_metric': 'auc',
        'task': 'classification',
        'cumulative_importance': 0.99
    })

    fs.plot_feature_importances(threshold=0.99, plot_n=10)
    selected = fs.remove(methods='all')

    X = selected

    # сокращение числа параметров


    X = RFE(X, Y)

    X['status'] = Y

    pandas.DataFrame([
        list(X)[:-1],
        X[X['status'] == 0].max().to_list()[:-1],
        X[X['status'] == 0].min().to_list()[:-1],
        X[X['status'] == 0].mean().to_list()[:-1]
    ]).T.to_csv("data/datasets/OUTPUT/legit_stats.csv", index=False,
                header=['feature', 'max', 'min', 'mean'])

    pandas.DataFrame([
        list(X)[:-1],
        X[X['status'] == 1].max().to_list()[:-1],
        X[X['status'] == 1].min().to_list()[:-1],
        X[X['status'] == 1].mean().to_list()[:-1]
    ]).T.to_csv("data/datasets/OUTPUT/phish_stats.csv", index=False,
                header=['feature', 'max', 'min', 'mean'])

    X.to_csv("data/datasets/OUTPUT/dataset.csv", index=False)


########################################################################################################################
#                                          Text segmentation
########################################################################################################################


def segment(text):
    return word_splitter.split(text)


########################################################################################################################
#               Uses https protocol
########################################################################################################################


def https_token(scheme):
    if scheme == 'https':
        return 0
    return 1


########################################################################################################################
#               Check if TLD in bad position
########################################################################################################################


def tld_in_path(tld, path):
    if path.count(tld) > 0:
        return 1
    return 0


def tld_in_subdomain(tld, subdomain):
    if subdomain.count(tld) > 0:
        return 1
    return 0


def tld_in_bad_position(tld, subdomain, path):
    if tld_in_path(tld, path) == 1 or tld_in_subdomain(tld, subdomain) == 1:
        return 1
    return 0


########################################################################################################################
#               Number of redirection to different domains
########################################################################################################################


def count_external_redirection(page, domain):
    if len(page.history) == 0:
        return 0
    else:
        count = 0
        for i, response in enumerate(page.history):
            if domain not in urlparse(response.url).netloc.lower():
                count += 1
        return count


########################################################################################################################
#               Is the registered domain created with random characters
########################################################################################################################


def random_word(word):
    if word in words_filter:
        return 0
    return 1


###############################tld_in_path#########################################################################################
#               Presence of words with random characters
########################################################################################################################


def random_words(url_words):
    return sum([random_word(word) for word in url_words])


########################################################################################################################
#               domain in brand list
########################################################################################################################


def sld_in_brand(sld):
    if sld in brand_filter:
        return 1
    return 0


########################################################################################################################
#               count www in url words
########################################################################################################################


def count_www(url_words):
    count = 0
    for word in url_words:
        if not word.find('www') == -1:
            count += 1
    return count


########################################################################################################################
#               length of raw word list
########################################################################################################################


def length_word_raw(url_words):
    return len(url_words)


########################################################################################################################
#               count average word length in raw word list
########################################################################################################################


def average_word_length(url_words):
    if len(url_words) == 0:
        return 0
    return sum(len(word) for word in url_words) / len(url_words)


########################################################################################################################
#               longest word length in raw word list
########################################################################################################################


def longest_word_length(url_words):
    if len(url_words) == 0:
        return 0
    return max([len(word) for word in url_words])


########################################################################################################################
#               Domain recognized by WHOIS
########################################################################################################################


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


########################################################################################################################
#               Unable to get web traffic and page rank
########################################################################################################################


session = session()


def web_traffic(short_url):
    try:
        with req_links_locker:
            rank = BeautifulSoup(session.get("http://data.alexa.com/data?cli=10&dat=s&url=" + short_url, timeout=3).text,
                             "xml").find("REACH")['RANK']

        return min(int(rank) / 10000000, 1)
    except:
        return 1


OPR_key = open("OPR_key.txt").read()


def page_rank(domain):
    url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
    try:
        with req_links_locker:
            request = session.get(url, headers={'API-OPR': OPR_key}, timeout=3)
        result = request.json()
        result = result['response'][0]['page_rank_decimal']
        if result:
            return result
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

def get_cert(hostname):
    result = None
    try:
        with req_links_locker:
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

def valid_cert_period(cert):
    try:
        return (cert['notAfter'] - cert['notBefore']).days
    except:
        return 0


########################################################################################################################
#               DNS record
########################################################################################################################


def good_netloc(netloc):
    try:
        with req_links_locker:
            socket.gethostbyname(netloc)
        return 1
    except:
        return 0


########################################################################################################################
########################################################################################################################
#                                               HTML
########################################################################################################################
########################################################################################################################


def urls_ratio(urls, total_urls):
    if len(total_urls) == 0:
        return 0
    else:
        return len(urls) / len(total_urls)


########################################################################################################################
#               ratio url-list
########################################################################################################################


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


########################################################################################################################
#              Ratio static/dynamic html content
########################################################################################################################


def ratio_dynamic_html(s_html, d_html):
    if s_html:
        return max(0, min(1, len(d_html) / len(s_html)))
    else:
        return 0


########################################################################################################################
#              Ratio html content on js-code
########################################################################################################################


def ratio_js_on_html(html_context):
    if html_context:
        return len(get_html_from_js(remove_JScomments(html_context))) / len(html_context)
    else:
        return 0


########################################################################################################################
#              Amount of http request operations (popular)
########################################################################################################################


def count_io_commands(string):
    pattern = r"(\".*?\"|\'.*?\'|\`.*?\`)|" \
              r"((.(open|send)|$.(get|post|ajax|getJSON)|fetch|axios(|.(get|post|all))|getData)\s*\()"
    regex = compile(pattern, MULTILINE | DOTALL)

    count = 0

    for m in finditer(regex, string):
        if not m.group(2) and m.groups():
            count += 1

    return count


########################################################################################################################
#                       OCR
########################################################################################################################


def translate_image(obj):
    try:
        resp = session.get(obj[0], stream=True, timeout=1).raw
        image = asarray(bytearray(resp.read()), dtype="uint8")
        img = imdecode(image, IMREAD_COLOR)
        img = resize(img, None, fx=0.35, fy=0.35, interpolation=INTER_AREA)
        img = GaussianBlur(img, (5, 5), 0)

        return image_to_string(img, lang=obj[1])
    except:
        return ""


def image_to_text(img, lang):
    if not img:
        return ""

    try:
        lang = languages.get(alpha2=lang).bibliographic

        if 'eng' not in lang:
            lang = 'eng+' + lang

        docs = []

        with req_img_locker:
            try:
                with ThreadPoolExecutor(25) as executor:
                    for r in executor.map(translate_image, [(url, lang) for url in img], timeout=3):
                        if r:
                            docs.append(r)
                return "\n".join(docs)
            except TimeoutError:
                return "\n".join(docs)
    except:
        return ""


########################################################################################################################
#                   Relationship between image text and context
########################################################################################################################


def ratio_Txt(dynamic, static):
    total = len(static)

    if total:
        return min(len(dynamic) / total, 1)
    else:
        return 0


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
    global p_v

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
    IFrame = {'visible': [], 'invisible': [], 'null': []}

    Text = ''
    Title = ''

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
    def collector11():
        for i_frame in soup.find_all('iframe', width=True, height=True, frameborder=True):
            if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['frameborder'] == "0":
                IFrame['invisible'].append(i_frame)
            else:
                IFrame['visible'].append(i_frame)
        for i_frame in soup.find_all('iframe', width=True, height=True, border=True):
            if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['border'] == "0":
                IFrame['invisible'].append(i_frame)
            else:
                IFrame['visible'].append(i_frame)
        for i_frame in soup.find_all('iframe', width=True, height=True, style=True):
            if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['style'] == "border:none;":
                IFrame['invisible'].append(i_frame)
            else:
                IFrame['visible'].append(i_frame)

    def merge_scripts(script_lnks):
        docs = []

        def load_script(url):
            state, request = is_URL_accessible(url, 1)

            if state:
                request.encoding = 'utf-8'
                docs.append(request.text)

        try:
            with req_links_locker:
                with ThreadPoolExecutor(25) as executor:
                    res = executor.map(load_script, script_lnks, timeout=3)

                    for r in res:
                        if r:
                            docs.append(r)
            return "\n".join(docs)
        except TimeoutError:
            return "\n".join(docs)

    with ThreadPoolExecutor(14) as e:
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
        e.submit(collector11)
        internals_script_doc = e.submit(merge_scripts, SCRIPT['internals']).result()
        externals_script_doc = e.submit(merge_scripts, SCRIPT['externals']).result()
        Text = e.submit(soup.get_text).result().lower()

    try:
        Title = soup.title.string.lower()
    except:
        pass

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

    return Href, Link, Anchor, Media, Img, Form, CSS, Favicon, IFrame, SCRIPT, Title, Text, internals_script_doc.lower(), externals_script_doc.lower(), io_count


def extract_text_context_data(content):
    soup = BeautifulSoup(content, 'lxml')

    io_count = len(soup.find_all('textarea')) + len(soup.find_all('input', type=None))
    for io in soup.find_all('input', type=True):
        if io['type'] == 'text' or io['type'] == 'password' or io['type'] == 'search':
            io_count += 1

    return soup.get_text().lower(), io_count


def word_ratio(Text_words):
    if Text_words:
        return len(Counter(Text_words)) / len(Text_words)
    else:
        return 0


def count_links(len):
    return len


def count_words(len):
    return len


###############################


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
def count_sBrackets(url):
    return len(re.findall('[\[\]]', url))
def count_rBrackets(url):
    return len(re.findall('[()]', url))
def count_space(url):
    return url.count(' ') + url.count('%20')
def count_double_slash(url):
    return url.count('//')-1
def ratio_digits(url):
    return len(re.sub("[^0-9]", "", url)) / len(url)
def count_digits(url):
    return len(re.sub("[^0-9]", "", url))
def abnormal_subdomain(url):
    if re.search('(http[s]?://(w[w]?|\d))([w]?(\d|-))', url):
        return 1
    return 0
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
def punycode(url):
    if url.startswith("http://xn--") or url.startswith("http://xn--"):
        return 1
    else:
        return 0
def brand_in_path(words_raw_path):
    for word in words_raw_path:
        if word in brand_filter:
            return 1
    return 0
def port(url):
    if re.search(
            "^[a-z][a-z0-9+\-.]*://([a-z0-9\-._~%!$&'()*+,;=]+@)?([a-z0-9\-._~%]+|\[[a-z0-9\-._~%!$&'()*+,;=:]+\]):([0-9]+)",
            url):
        return 1
    return 0
def shortest_word_length(words_raw):
    if len(words_raw) == 0:
        return 0
    return min(len(word) for word in words_raw)
def prefix_suffix(url):
    if re.findall(r"https?://[^\-]+-[^\-]+/", url):
        return 1
    else:
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
        return requests.get(url, timeout=1)
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
def count_reqs_redirections(reqs):
    if len(reqs) > 0:
        count = 0
        for req in reqs:
            if req:
                if req.is_redirect:
                    count += 1

        return count / len(reqs)
    else:
        return 0
def count_reqs_error(reqs):
    if len(reqs) > 0:
        count = 0
        for req in reqs:
            if req:
                if req.status_code >= 400:
                    count += 1
            else:
                count += 1

        return count / len(reqs)
    else:
        return 0

def login_form(Form):
    if len(Form['externals']) > 0 or len(Form['null']) > 0:
        return 1

    p = re.compile('([a-zA-Z0-9\_])+.php')
    for form in Form['internals'] + Form['externals']:
        if p.match(form) != None:
            return 1
    return 0
def submitting_to_email(Form):
    for form in Form['internals'] + Form['externals']:
        if "mailto:" in form or "mail()" in form:
            return 1
    return 0
def empty_title(Title):
    if Title:
        return 0
    return 1
def iframe(IFrame):
    if len(IFrame['invisible']) > 0:
        return 1
    return 0
def onmouseover(content):
    if 'onmouseover="window.status=' in content.replace(" ", ""):
        return 1
    else:
        return 0
def popup_window(content):
    if "prompt(" in content:
        return 1
    else:
        return 0
def right_clic(content):
    if re.findall(r"event.button\s*==\s*2", content):
        return 1
    else:
        return 0

def domain_in_text(second_level_domain, text):
    if second_level_domain in text:
        return 1
    return 0
def domain_with_copyright(domain, content):
    try:
        m = re.search(u'(\N{COPYRIGHT SIGN}|\N{TRADE MARK SIGN}|\N{REGISTERED SIGN})', content)
        _copyright = content[m.span()[0] - 50:m.span()[0] + 50]
        if domain in _copyright:
            return 1
        else:
            return 0
    except:
        return 0
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

def remainder_valid_cert(cert):
    try:
        period = cert['notAfter'] - cert['notBefore']

        today = time.strftime('%Y-%m-%d')
        today = datetime.strptime(today, '%Y-%m-%d')

        passed_time = today - cert['notBefore']

        return max(0, min(1, passed_time / period))
    except:
        return 0

##############################


########################################################################################################################
#               visual similarity of symbols
########################################################################################################################


symbols = {
    'g': ['g', 'q'],
    'q': ['q', 'g'],
    'l': ['l', '1', 'i'],
    '1': ['1', 'l', 'i'],
    'i': ['i', '1', 'l'],
    'o': ['o', '0'],
    '0': ['0', 'o'],
    'rn': ['rn', 'm'],
    'vv': ['vv', 'w'],
    'm': ['m', 'rn'],
    'w': ['w', 'vv']
}


from functools import reduce
from operator import mul


def count_visual_similarity_domains(second_level_domain):
    finded = {}

    # поиск заменяемых символов
    for s, lst in symbols.items():
        shift = 0
        while True:
            idx = second_level_domain.find(s, shift)

            if idx < 0:
                break

            shift = idx + 1
            finded[idx] = {'from': s, 'to': lst}

    dic = []
    pattern = second_level_domain

    # формирование паттерна
    for key in sorted(finded.keys()):
        pattern = pattern.replace(finded[key]['from'], '{}')
        dic.append(finded[key]['to'])

    N = len(dic)
    dividers = [reduce(mul, [len(dic[N - y - 1]) for y in range(N - x - 1)], 1) for x in range(N)]      # создание границ списка (для поиска)
    max_size = reduce(mul, [len(dic[x]) for x in range(N)], 1) - 1      # кол-во возможных комбинаций
    samples = [pattern.format(*[dic[x][int((i + 1) / dividers[x]) % len(dic[x])] for x in range(N)]) for i in range(max_size)]      # генерация примеров слова с заменяемыми символами

    count = 0

    for s in samples:
        if s in brand_filter:
            count += 1

    return count

###################################


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
            parsed = urlparse(r_url)
            tld = extracted_domain.suffix
            scheme = parsed.scheme

            with ThreadPoolExecutor(2) as e:
                cert = e.submit(get_cert, domain).result()
                (Href, Link, Anchor, Media, Img, Form, CSS, Favicon, IFrame, SCRIPT, Title, Text, internals_script_doc,
                 externals_script_doc,
                 count_textareas) = e.submit(
                    extract_all_context_data, hostname, content, domain, r_url).result()

            content_di = get_html_from_js(remove_JScomments(internals_script_doc))
            content_de = get_html_from_js(remove_JScomments(externals_script_doc))

            with ThreadPoolExecutor(2) as e:
                (Href_di, Link_di, Anchor_di, Media_di, Img_di, Form_di, CSS_di, Favicon_di, IFrame_di, SCRIPT_di,
                 Title_di,
                 Text_di, _, _, count_textareas_di) = e.submit(
                    extract_all_context_data, hostname, content_di, domain, r_url).result()
                (Href_de, Link_de, Anchor_de, Media_de, Img_de, Form_de, CSS_de, Favicon_de, IFrame_de, SCRIPT_de,
                 Title_de,
                 Text_de, _, _, count_textareas_de) = e.submit(
                    extract_all_context_data, hostname, content_di, domain, r_url).result()

            # Text_di, count_textareas_di = extract_text_context_data(content_di)
            # Text_de, count_textareas_de = extract_text_context_data(content_de)

            with req_links_locker:
                lang = check_Language(Text)

            with ThreadPoolExecutor(2) as e:
                internals_img_txt = e.submit(image_to_text, Img['internals'], lang).result().lower()
                externals_img_txt = e.submit(image_to_text, Img['externals'], lang).result().lower()

            iImgTxt_words = reg.findall(internals_img_txt)
            eImgTxt_words = reg.findall(externals_img_txt)

            sContent_words = reg.findall(Text)
            diContent_words = reg.findall(Text_di)
            deContent_words = reg.findall(Text_de)

            Text_words = iImgTxt_words + eImgTxt_words + sContent_words + diContent_words + deContent_words

            iUrl_s = Href['internals'] + Link['internals'] + Media['internals'] + Form['internals']
            eUrl_s = Href['externals'] + Link['externals'] + Media['externals'] + Form['externals']
            nUrl_s = Href['null'] + Link['null'] + Media['null'] + Form['null']

            with req_links_locker:
                with ThreadPoolExecutor(2) as e:
                    reqs_iData_s = e.submit(get_reqs_data, iUrl_s).result()
                    reqs_eData_s = e.submit(get_reqs_data, eUrl_s).result()

            iUrl_di = Href_di['internals'] + Link_di['internals'] + Media_di['internals'] + Form_di['internals']
            eUrl_di = Href_di['externals'] + Link_di['externals'] + Media_di['externals'] + Form_di['externals']
            nUrl_di = Href_di['null'] + Link_di['null'] + Media_di['null'] + Form_di['null']

            with req_links_locker:
                with ThreadPoolExecutor(2) as e:
                    reqs_iData_di = e.submit(get_reqs_data, iUrl_di).result()
                    reqs_eData_di = e.submit(get_reqs_data, eUrl_di).result()

            iUrl_de = Href_de['internals'] + Link_de['internals'] + Media_de['internals'] + Form_de['internals']
            eUrl_de = Href_de['externals'] + Link_de['externals'] + Media_de['externals'] + Form_de['externals']
            nUrl_de = Href_de['null'] + Link_de['null'] + Media_de['null'] + Form_de['null']

            with req_links_locker:
                with ThreadPoolExecutor(2) as e:
                    reqs_iData_de = e.submit(get_reqs_data, iUrl_de).result()
                    reqs_eData_de = e.submit(get_reqs_data, eUrl_de).result()

                whois_domain = whois.whois(domain)

            result = []

            with ThreadPoolExecutor(190) as e:
                result.append(e.submit(word_ratio, Text_words).result())
                result.append(e.submit(having_ip_address, url).result())
                result.append(e.submit(shortening_service, url).result())
                result.append(int(cert != None))
                result.append(e.submit(good_netloc, netloc).result())
                result.append(len(r_url))
                result.append(e.submit(r_url.count, '@').result())
                result.append(e.submit(r_url.count, '!').result())
                result.append(e.submit(r_url.count, '+').result())
                result.append(e.submit(count_sBrackets, r_url).result())
                result.append(e.submit(count_rBrackets, r_url).result())
                result.append(e.submit(r_url.count, ',').result())
                result.append(e.submit(r_url.count, '$').result())
                result.append(e.submit(r_url.count, ';').result())
                result.append(e.submit(count_space, r_url).result())
                result.append(e.submit(r_url.count, '&').result())
                result.append(e.submit(count_double_slash, r_url).result())
                result.append(e.submit(r_url.count, '/').result() - 2)
                result.append(e.submit(r_url.count, '=').result())
                result.append(e.submit(r_url.count, '%').result())
                result.append(e.submit(r_url.count, '?').result())
                result.append(e.submit(r_url.count, '_').result())
                result.append(e.submit(r_url.count, '-').result())
                result.append(e.submit(r_url.count, '.').result())
                result.append(e.submit(r_url.count, ':').result())
                result.append(e.submit(r_url.count, '*').result())
                result.append(e.submit(r_url.count, '|').result())
                result.append(e.submit(r_url.count, '~').result())
                result.append(e.submit(r_url.count, 'http').result())
                result.append(e.submit(https_token, scheme).result())
                result.append(e.submit(ratio_digits, r_url).result())
                result.append(e.submit(count_digits, r_url).result())
                result.append(e.submit(count_phish_hints, url_words, phish_hints, lang).result())
                result.append(len(url_words))
                result.append(e.submit(tld_in_path, tld, path).result())
                result.append(e.submit(tld_in_subdomain, tld, subdomain).result())
                result.append(e.submit(tld_in_bad_position, tld, subdomain, path).result())
                result.append(e.submit(abnormal_subdomain, r_url).result())
                result.append(len(request.history))
                result.append(e.submit(count_external_redirection, request, domain).result())
                result.append(e.submit(random_word, second_level_domain).result())
                result.append(e.submit(random_words, url_words).result())
                result.append(e.submit(random_words, words_raw_host).result())
                result.append(e.submit(random_words, words_raw_path).result())
                result.append(e.submit(char_repeat, url_words).result())
                result.append(e.submit(char_repeat, words_raw_host).result())
                result.append(e.submit(char_repeat, words_raw_path).result())
                result.append(e.submit(punycode, r_url).result())
                result.append(e.submit(sld_in_brand, second_level_domain).result())
                result.append(e.submit(brand_in_path, words_raw_path).result())
                result.append(e.submit(cutted_url.count, 'www').result())
                result.append(e.submit(cutted_url.count, 'com').result())
                result.append(e.submit(port, r_url).result())
                result.append(e.submit(length_word_raw, url_words).result())            # todo: ???
                result.append(e.submit(average_word_length, url_words).result())
                result.append(e.submit(longest_word_length, url_words).result())
                result.append(e.submit(shortest_word_length, url_words).result())
                result.append(e.submit(prefix_suffix, r_url).result())
                result.append(e.submit(count_subdomain, netloc).result())
                result.append(e.submit(count_visual_similarity_domains, second_level_domain).result())
                result.append(e.submit(compression_ratio, request).result())
                result.append(count_textareas)
                result.append(e.submit(ratio_js_on_html, Text).result())
                result.append(len(iUrl_s) + len(eUrl_s))
                result.append(e.submit(urls_ratio, iUrl_s, iUrl_s + eUrl_s + nUrl_s).result())
                result.append(e.submit(urls_ratio, eUrl_s, iUrl_s + eUrl_s + nUrl_s).result())
                result.append(e.submit(urls_ratio, nUrl_s, iUrl_s + eUrl_s + nUrl_s).result())
                result.append(e.submit(ratio_List, CSS, 'internals').result())
                result.append(e.submit(ratio_List, CSS, 'externals').result())
                result.append(e.submit(ratio_List, CSS, 'embedded').result())
                result.append(e.submit(ratio_List, SCRIPT, 'internals').result())
                result.append(e.submit(ratio_List, SCRIPT, 'externals').result())
                result.append(e.submit(ratio_List, SCRIPT, 'embedded').result())
                result.append(e.submit(ratio_List, Img, 'externals').result())
                result.append(e.submit(ratio_List, Img, 'internals').result())
                result.append(e.submit(count_reqs_redirections, reqs_iData_s).result())
                result.append(e.submit(count_reqs_redirections, reqs_eData_s).result())
                result.append(e.submit(count_reqs_error, reqs_iData_s).result())
                result.append(e.submit(count_reqs_error, reqs_eData_s).result())
                result.append(e.submit(login_form, Form).result())
                result.append(e.submit(ratio_List, Favicon, 'externals').result())
                result.append(e.submit(ratio_List, Favicon, 'internals').result())
                result.append(e.submit(submitting_to_email, Form).result())
                result.append(e.submit(ratio_List, Media, 'internals').result())
                result.append(e.submit(ratio_List, Media, 'externals').result())
                result.append(e.submit(empty_title, Title).result())
                result.append(e.submit(ratio_anchor, Anchor, 'unsafe').result())
                result.append(e.submit(ratio_anchor, Anchor, 'safe').result())
                result.append(e.submit(ratio_List, Link, 'internals').result())
                result.append(e.submit(ratio_List, Link, 'externals').result())
                result.append(e.submit(iframe, IFrame).result())
                result.append(e.submit(onmouseover, content).result())
                result.append(e.submit(popup_window, content).result())
                result.append(e.submit(right_clic, content).result())
                result.append(e.submit(domain_in_text, second_level_domain, Text).result())
                result.append(e.submit(domain_in_text, second_level_domain, Title).result())
                result.append(e.submit(domain_with_copyright, domain, content).result())
                result.append(e.submit(count_phish_hints, Text, phish_hints, lang).result())
                result.append(len(sContent_words))
                result.append(e.submit(ratio_Txt, iImgTxt_words + eImgTxt_words, sContent_words).result())
                result.append(e.submit(ratio_Txt, iImgTxt_words, sContent_words).result())
                result.append(e.submit(ratio_Txt, eImgTxt_words, sContent_words).result())
                result.append(e.submit(ratio_Txt, eImgTxt_words, iImgTxt_words).result())
                result.append(e.submit(ratio_dynamic_html, Text, "".join([Text_di, Text_de])).result())
                result.append(e.submit(ratio_dynamic_html, Text, Text_di).result())
                result.append(e.submit(ratio_js_on_html, Text_di).result())
                result.append(count_textareas_di)
                result.append(len(iUrl_di) + len(eUrl_di))
                result.append(e.submit(urls_ratio, iUrl_di, iUrl_di + eUrl_di + nUrl_di + nUrl_di).result())
                result.append(e.submit(urls_ratio, eUrl_di, iUrl_di + eUrl_di + nUrl_di).result())
                result.append(e.submit(urls_ratio, nUrl_di, iUrl_di + eUrl_di + nUrl_di).result())
                result.append(e.submit(ratio_List, CSS_di, 'internals').result())
                result.append(e.submit(ratio_List, CSS_di, 'externals').result())
                result.append(e.submit(ratio_List, CSS_di, 'embedded').result())
                result.append(e.submit(ratio_List, SCRIPT_di, 'internals').result())
                result.append(e.submit(ratio_List, SCRIPT_di, 'externals').result())
                result.append(e.submit(ratio_List, SCRIPT_di, 'embedded').result())
                result.append(e.submit(ratio_List, Img_di, 'externals').result())
                result.append(e.submit(ratio_List, Img_di, 'internals').result())
                result.append(e.submit(count_reqs_redirections, reqs_iData_di).result())
                result.append(e.submit(count_reqs_redirections, reqs_eData_di).result())
                result.append(e.submit(count_reqs_error, reqs_iData_di).result())
                result.append(e.submit(count_reqs_error, reqs_eData_di).result())
                result.append(e.submit(login_form, Form_di).result())
                result.append(e.submit(ratio_List, Favicon_di, 'externals').result())
                result.append(e.submit(ratio_List, Favicon_di, 'internals').result())
                result.append(e.submit(submitting_to_email, Form_di).result())
                result.append(e.submit(ratio_List, Media_di, 'internals').result())
                result.append(e.submit(ratio_List, Media_di, 'externals').result())
                result.append(e.submit(empty_title, Title_di).result())
                result.append(e.submit(ratio_anchor, Anchor_di, 'unsafe').result())
                result.append(e.submit(ratio_anchor, Anchor_di, 'safe').result())
                result.append(e.submit(ratio_List, Link_di, 'internals').result())
                result.append(e.submit(ratio_List, Link_di, 'externals').result())
                result.append(e.submit(iframe, IFrame_di).result())
                result.append(e.submit(onmouseover, content_di).result())
                result.append(e.submit(popup_window, content_di).result())
                result.append(e.submit(right_clic, content_di).result())
                result.append(e.submit(domain_in_text, second_level_domain, Text_di).result())
                result.append(e.submit(domain_in_text, second_level_domain, Title_di).result())
                result.append(e.submit(domain_with_copyright, domain, content_di).result())
                result.append(e.submit(count_io_commands, internals_script_doc).result())
                result.append(e.submit(count_phish_hints, Text_di, phish_hints, lang).result())
                result.append(len(diContent_words))
                result.append(e.submit(ratio_dynamic_html, Text, Text_de).result())
                result.append(e.submit(ratio_js_on_html, Text_de).result())
                result.append(count_textareas_de)
                result.append(len(iUrl_de) + len(eUrl_de))
                result.append(e.submit(urls_ratio, iUrl_de, iUrl_de + eUrl_de + nUrl_de).result())
                result.append(e.submit(urls_ratio, eUrl_de, iUrl_de + eUrl_de + nUrl_de).result())
                result.append(e.submit(urls_ratio, nUrl_de, iUrl_de + eUrl_de + nUrl_de).result())
                result.append(e.submit(ratio_List, CSS_de, 'internals').result())
                result.append(e.submit(ratio_List, CSS_de, 'externals').result())
                result.append(e.submit(ratio_List, CSS_de, 'embedded').result())
                result.append(e.submit(ratio_List, SCRIPT_de, 'internals').result())
                result.append(e.submit(ratio_List, SCRIPT_de, 'externals').result())
                result.append(e.submit(ratio_List, SCRIPT_de, 'embedded').result())
                result.append(e.submit(ratio_List, Img_de, 'externals').result())
                result.append(e.submit(ratio_List, Img_de, 'internals').result())
                result.append(e.submit(count_reqs_redirections, reqs_iData_de).result())
                result.append(e.submit(count_reqs_redirections, reqs_eData_de).result())
                result.append(e.submit(count_reqs_error, reqs_iData_de).result())
                result.append(e.submit(count_reqs_error, reqs_eData_de).result())
                result.append(e.submit(login_form, Form_de).result())
                result.append(e.submit(ratio_List, Favicon_de, 'externals').result())
                result.append(e.submit(ratio_List, Favicon_de, 'internals').result())
                result.append(e.submit(submitting_to_email, Form_de).result())
                result.append(e.submit(ratio_List, Media_de, 'internals').result())
                result.append(e.submit(ratio_List, Media_de, 'externals').result())
                result.append(e.submit(empty_title, Title_de).result())
                result.append(e.submit(ratio_anchor, Anchor_de, 'unsafe').result())
                result.append(e.submit(ratio_anchor, Anchor_de, 'safe').result())
                result.append(e.submit(ratio_List, Link_de, 'internals').result())
                result.append(e.submit(ratio_List, Link_de, 'externals').result())
                result.append(e.submit(iframe, IFrame_de).result())
                result.append(e.submit(onmouseover, content_de).result())
                result.append(e.submit(popup_window, content_de).result())
                result.append(e.submit(right_clic, content_de).result())
                result.append(e.submit(domain_in_text, second_level_domain, Text_de).result())
                result.append(e.submit(domain_in_text, second_level_domain, Title_de).result())
                result.append(e.submit(domain_with_copyright, domain, content_de).result())
                result.append(e.submit(count_io_commands, externals_script_doc).result())
                result.append(e.submit(count_phish_hints, Text_de, phish_hints, lang).result())
                result.append(len(deContent_words))
                result.append(e.submit(domain_expiration, whois_domain).result())
                result.append(e.submit(whois_registered_domain, whois_domain, domain).result())
                result.append(e.submit(web_traffic, r_url).result())
                result.append(e.submit(page_rank, domain).result())
                result.append(e.submit(remainder_valid_cert, cert).result())
                result.append(e.submit(valid_cert_period, cert).result())
                result.append(e.submit(count_alt_names, cert).result())

            return result
        return request
    except Exception as ex:
        print(ex)
        return ex


#
# import pandas as pd
# import os
# import csv
# import pickle
# import homoglyphs as hg
#
# project_path = '../FrequencyWords/content'
#
# df = pd.DataFrame()
#
# for folder in os.listdir(project_path):
#     for lang in os.listdir(project_path+'/'+folder):
#         for file in os.listdir(project_path+'/'+folder+'/'+lang):
#             if file.endswith('_50k.txt'):
#                 df = pd.read_csv(project_path + '/' + folder + '/' + lang + '/' + file,
#                                            sep=' ', error_bad_lines=False, low_memory=False,
#                                            quoting=csv.QUOTE_NONE, encoding='utf-8', header=None)
#
#                 df.to_csv('vocabulary50xN.csv', sep=' ', header=False, columns=None, mode='a', index=False)
#                 print(folder, lang)
#
# w = pickle.load(open('Trie.pkl', 'rb'))
#
# df = pd.read_csv('vocabulary50xN.csv.gz', sep=' ', error_bad_lines=False, low_memory=False, quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, compression='gzip')
# df = df.sort_values(by=1, ascending=False)
# df = df.drop_duplicates(ignore_index=False, subset=0)
# df[0].to_csv('vocabulary50xN2.csv.gz', sep=' ', header=False, columns=None, index=False, compression='gzip')
#
# from tqdm import tqdm
# from bloomfpy import BloomFilter
# import pickle
#
# f = BloomFilter(capacity=len(df), error_rate=0.001)
#
# for w in tqdm(df[0].values, total=len(df)):
#     f.add(w)
#
#
# pickle.dump(f, open('Trie.pkl', 'wb'))
#
# f = pickle.load(open('Trie.pkl', 'rb'))
#
#
#
# def insert_row(idx, df, df_insert):
#     return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ]).reset_index(drop = True)
#
#
# for i,v in enumerate(df[0].values):
#     if v in 'м':
#         print(i,v)
#
# rate_lang = {}
# folder = "2016"
# for lang in os.listdir(project_path+'/'+folder):
#     for file in os.listdir(project_path+'/'+folder+'/'+lang):
#         if file.endswith('_full.txt'):
#             df = pd.read_csv(project_path + '/' + folder + '/' + lang + '/' + file,
#                                        sep=' ', error_bad_lines=False, low_memory=False,
#                                        quoting=csv.QUOTE_NONE, encoding='utf-8', header=None)
#             print(lang, len(df))
#
# lang_df = pd.DataFrame(rate_lang, index=[0]).T.sort_values(by=0,ascending=False)
#
# def rem(lang, df, n):
#     count = 0
#     try:
#         for c in hg.Languages.get_alphabet([lang]):
#             try:
#                 idx = df[df[0] == c].index[0]
#                 if idx > n:
#                     df = df.drop(idx)
#                     count+=1
#             except:
#                 pass
#     except:
#         pass
#     return df
#
# for folder in os.listdir(project_path):
#     for lang in os.listdir(project_path+'/'+folder):
#         for file in os.listdir(project_path+'/'+folder+'/'+lang):
#             if file.endswith('_50k.txt'):
#                 df = pd.read_csv(project_path + '/' + folder + '/' + lang + '/' + file,
#                                            sep=' ', error_bad_lines=False, low_memory=False,
#                                            quoting=csv.QUOTE_NONE, encoding='utf-8', header=None)
#                 l1 = len(df)
#                 df = rem(lang, df, 50)
#                 l2 = len(df)
#                 df.to_csv('w.csv', sep=' ', header=False, columns=None, mode='a', index=False)
#                 print(folder, lang, l1-l2)
#
# df = pd.DataFrame()
#
# for folder in os.listdir(project_path):
#     for lang in ['hu', 'fi', 'tr', 'sr', 'ar', 'cs', 'hr', 'el', 'pl', 'ru', 'en', 'es', 'bg', 'bs', 'he', 'et', 'ro', 'nl', 'pt_br', 'sl', 'de', 'sk', 'sv', 'it', 'fr', 'da', 'pt', 'zh', 'no', 'ko', 'zh_tw', 'lt', 'fa', 'mk', 'is', 'uk', 'sq', 'id', 'lv', 'ja', 'th', 'ka', 'gl', 'ms', 'ca', 'vi', 'eu', 'eo', 'af', 'si', 'br', 'tl', 'kk', 'hi', 'ml', 'bn', 'hy', 'te', 'ta']:
#         try:
#             for file in os.listdir(project_path+'/'+folder+'/'+lang):
#                 if file.endswith('_50k.txt'):
#                     df = pd.read_csv(project_path + '/' + folder + '/' + lang + '/' + file,
#                                                sep=' ', error_bad_lines=False, low_memory=False,
#                                                quoting=csv.QUOTE_NONE, encoding='utf-8', header=None)
#
#                     l1 = len(df)
#                     df = rem(lang, df, 50)
#                     l2 = len(df)
#                     df.to_csv('w.csv', sep=' ', header=False, columns=None, mode='a', index=False)
#                     print(folder, lang, l1 - l2)
#         except:
#             pass
#
# for folder in os.listdir(project_path):
#     for lang in os.listdir(project_path+'/'+folder):
#         print(lang, rem(lang))
#
#
# ph = pickle.load(open('phish_hints.pkl','rb'))
#
# for k,hints in ph.items():
#     df = insert_row(800000, df, pd.DataFrame(hints))
#
#
# def num_there(s):
#     return any(i.isdigit() for i in s)
#
# brands = pd.read_csv('brands.csv', sep=' ', error_bad_lines=False, low_memory=False, quoting=csv.QUOTE_NONE, encoding='utf-8', header=None)
# br = []
#
# for b in brands[0].values:
#     if num_there(b):
#         continue
#
#     bb = b.replace('-','').split('.')
#
#     if len(bb) == 2:
#         br.append(bb[0])
#     else:
#         br.append(''.join(bb[:-1]))
#
#
# df = df[~df[0].isin(brands[0].values)]
