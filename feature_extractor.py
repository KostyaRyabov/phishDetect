import concurrent.futures
from googletrans import Translator
from console_progressbar import ProgressBar
from datetime import date
import pandas
import tldextract
import requests
import os
import re
from random import randint
from tools import benchmark, tokenize, segment

from urllib.parse import urlparse, urlsplit
from bs4 import BeautifulSoup

import url_features as uf
import external_features as ef


key = open("OPR_key.txt").read()

translator = Translator()


def load_phishHints():
    hints_dir = "data/phish_hints/"
    file_list = os.listdir(hints_dir)

    if file_list:
        return [{leng[0:2]: pandas.read_csv(hints_dir + leng, header=None)[0].tolist()} for leng in file_list][0]
    else:
        hints = {'en': [
            'login',
            'logon',
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


phish_hints = load_phishHints()

@benchmark
def check_Language(text):
    language = translator.detect(text).lang

    if language not in phish_hints:
        phish_hints[language] = translator.translate(";".join(phish_hints['en'][0:16]), src='en',
                                                     dest=language).text.split(";")
        data = pandas.DataFrame(phish_hints[language])
        filename = "data/phish_hints/{0}.csv".format(language)
        data.to_csv(filename, index=False, header=False)

    return language


@benchmark
def is_URL_accessible(url, time_out=5):
    page = None
    try:
        page = requests.get(url, timeout=time_out)
    except:
        parsed = urlparse(url)
        url = parsed.scheme + '://' + parsed.netloc
        if not parsed.netloc.startswith('www'):
            url = parsed.scheme + '://www.' + parsed.netloc
            try:
                page = requests.get(url, timeout=time_out)
            except:
                page = None
                pass

    if page and page.status_code == 200 and page.content not in ["b''", "b' '"]:
        return True, page
    else:
        return False, None


def get_domain(url):
    o = urlsplit(url)
    return o.hostname, tldextract.extract(url).domain, o.path

@benchmark
def extract_data_from_URL(hostname, content, domain):
    Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                   "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

    Href = {'internals': [], 'externals': [], 'null': []}
    Link = {'internals': [], 'externals': [], 'null': []}
    Anchor = {'safe': [], 'unsafe': [], 'null': []}
    Media = {'internals': [], 'externals': [], 'null': []}
    Form = {'internals': [], 'externals': [], 'null': []}
    CSS = {'internals': [], 'externals': [], 'null': []}
    Favicon = {'internals': [], 'externals': [], 'null': []}
    IFrame = {'visible': [], 'invisible': [], 'null': []}
    Title = ''
    Text = ''

    soup = BeautifulSoup(content, 'html.parser', from_encoding='iso-8859-1')

    # collect all external and internal hrefs from url
    for href in soup.find_all('a', href=True):
        dots = [x.start(0) for x in re.finditer('\.', href['href'])]
        if hostname in href['href'] or domain in href['href'] or len(dots) == 1 or not href['href'].startswith('http'):
            if "#" in href['href'] or "javascript" in href['href'].lower() or "mailto" in href['href'].lower():
                Anchor['unsafe'].append(href['href'])
            if not href['href'].startswith('http'):
                if not href['href'].startswith('/'):
                    Href['internals'].append(hostname + '/' + href['href'])
                elif href['href'] in Null_format:
                    Href['null'].append(href['href'])
                else:
                    Href['internals'].append(hostname + href['href'])
        else:
            Href['externals'].append(href['href'])
            Anchor['safe'].append(href['href'])

    # collect all media src tags
    for img in soup.find_all('img', src=True):
        dots = [x.start(0) for x in re.finditer('\.', img['src'])]
        if hostname in img['src'] or domain in img['src'] or len(dots) == 1 or not img['src'].startswith('http'):
            if not img['src'].startswith('http'):
                if not img['src'].startswith('/'):
                    Media['internals'].append(hostname + '/' + img['src'])
                elif img['src'] in Null_format:
                    Media['null'].append(img['src'])
                else:
                    Media['internals'].append(hostname + img['src'])
        else:
            Media['externals'].append(img['src'])

    for audio in soup.find_all('audio', src=True):
        dots = [x.start(0) for x in re.finditer('\.', audio['src'])]
        if hostname in audio['src'] or domain in audio['src'] or len(dots) == 1 or not audio['src'].startswith('http'):
            if not audio['src'].startswith('http'):
                if not audio['src'].startswith('/'):
                    Media['internals'].append(hostname + '/' + audio['src'])
                elif audio['src'] in Null_format:
                    Media['null'].append(audio['src'])
                else:
                    Media['internals'].append(hostname + audio['src'])
        else:
            Media['externals'].append(audio['src'])

    for embed in soup.find_all('embed', src=True):
        dots = [x.start(0) for x in re.finditer('\.', embed['src'])]
        if hostname in embed['src'] or domain in embed['src'] or len(dots) == 1 or not embed['src'].startswith('http'):
            if not embed['src'].startswith('http'):
                if not embed['src'].startswith('/'):
                    Media['internals'].append(hostname + '/' + embed['src'])
                elif embed['src'] in Null_format:
                    Media['null'].append(embed['src'])
                else:
                    Media['internals'].append(hostname + embed['src'])
        else:
            Media['externals'].append(embed['src'])

    for i_frame in soup.find_all('iframe', src=True):
        dots = [x.start(0) for x in re.finditer('\.', i_frame['src'])]
        if hostname in i_frame['src'] or domain in i_frame['src'] or len(dots) == 1 or not i_frame['src'].startswith(
                'http'):
            if not i_frame['src'].startswith('http'):
                if not i_frame['src'].startswith('/'):
                    Media['internals'].append(hostname + '/' + i_frame['src'])
                elif i_frame['src'] in Null_format:
                    Media['null'].append(i_frame['src'])
                else:
                    Media['internals'].append(hostname + i_frame['src'])
        else:
            Media['externals'].append(i_frame['src'])

    # collect all link tags
    for link in soup.findAll('link', href=True):
        dots = [x.start(0) for x in re.finditer('\.', link['href'])]
        if hostname in link['href'] or domain in link['href'] or len(dots) == 1 or not link['href'].startswith('http'):
            if not link['href'].startswith('http'):
                if not link['href'].startswith('/'):
                    Link['internals'].append(hostname + '/' + link['href'])
                elif link['href'] in Null_format:
                    Link['null'].append(link['href'])
                else:
                    Link['internals'].append(hostname + link['href'])
        else:
            Link['externals'].append(link['href'])

    for script in soup.find_all('script', src=True):
        dots = [x.start(0) for x in re.finditer('\.', script['src'])]
        if hostname in script['src'] or domain in script['src'] or len(dots) == 1 or not script['src'].startswith(
                'http'):
            if not script['src'].startswith('http'):
                if not script['src'].startswith('/'):
                    Link['internals'].append(hostname + '/' + script['src'])
                elif script['src'] in Null_format:
                    Link['null'].append(script['src'])
                else:
                    Link['internals'].append(hostname + script['src'])
        else:
            Link['externals'].append(link['href'])

    # collect all css
    for link in soup.find_all('link', rel='stylesheet'):
        dots = [x.start(0) for x in re.finditer('\.', link['href'])]
        if hostname in link['href'] or domain in link['href'] or len(dots) == 1 or not link['href'].startswith('http'):
            if not link['href'].startswith('http'):
                if not link['href'].startswith('/'):
                    CSS['internals'].append(hostname + '/' + link['href'])
                elif link['href'] in Null_format:
                    CSS['null'].append(link['href'])
                else:
                    CSS['internals'].append(hostname + link['href'])
        else:
            CSS['externals'].append(link['href'])

    for style in soup.find_all('style', type='text/css'):
        try:
            start = str(style[0]).index('@import url(')
            end = str(style[0]).index(')')
            css = str(style[0])[start + 12:end]
            dots = [x.start(0) for x in re.finditer('\.', css)]
            if hostname in css or domain in css or len(dots) == 1 or not css.startswith('http'):
                if not css.startswith('http'):
                    if not css.startswith('/'):
                        CSS['internals'].append(hostname + '/' + css)
                    elif css in Null_format:
                        CSS['null'].append(css)
                    else:
                        CSS['internals'].append(hostname + css)
            else:
                CSS['externals'].append(css)
        except:
            continue

    # collect all form actions
    for form in soup.findAll('form', action=True):
        dots = [x.start(0) for x in re.finditer('\.', form['action'])]
        if hostname in form['action'] or domain in form['action'] or len(dots) == 1 or not form['action'].startswith(
                'http'):
            if not form['action'].startswith('http'):
                if not form['action'].startswith('/'):
                    Form['internals'].append(hostname + '/' + form['action'])
                elif form['action'] in Null_format or form['action'] == 'about:blank':
                    Form['null'].append(form['action'])
                else:
                    Form['internals'].append(hostname + form['action'])
        else:
            Form['externals'].append(form['action'])

    # collect all link tags
    for head in soup.find_all('head'):
        for head.link in soup.find_all('link', href=True):
            dots = [x.start(0) for x in re.finditer('\.', head.link['href'])]
            if hostname in head.link['href'] or len(dots) == 1 or domain in head.link['href'] or not head.link[
                'href'].startswith('http'):
                if not head.link['href'].startswith('http'):
                    if not head.link['href'].startswith('/'):
                        Favicon['internals'].append(hostname + '/' + head.link['href'])
                    elif head.link['href'] in Null_format:
                        Favicon['null'].append(head.link['href'])
                    else:
                        Favicon['internals'].append(hostname + head.link['href'])
            else:
                Favicon['externals'].append(head.link['href'])

        for head.link in soup.findAll('link', {'href': True, 'rel': True}):
            isicon = False
            if isinstance(head.link['rel'], list):
                for e_rel in head.link['rel']:
                    if (e_rel.endswith('icon')):
                        isicon = True
            else:
                if (head.link['rel'].endswith('icon')):
                    isicon = True

            if isicon:
                dots = [x.start(0) for x in re.finditer('\.', head.link['href'])]
                if hostname in head.link['href'] or len(dots) == 1 or domain in head.link['href'] or not head.link[
                    'href'].startswith('http'):
                    if not head.link['href'].startswith('http'):
                        if not head.link['href'].startswith('/'):
                            Favicon['internals'].append(hostname + '/' + head.link['href'])
                        elif head.link['href'] in Null_format:
                            Favicon['null'].append(head.link['href'])
                        else:
                            Favicon['internals'].append(hostname + head.link['href'])
                else:
                    Favicon['externals'].append(head.link['href'])

    # collect i_frame
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

    # get page title
    try:
        Title = soup.title.string
    except:
        pass

    # get content text
    Text = soup.get_text()

    return Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text


def find_duplicates(list):
    class Dictlist(dict):
        def __setitem__(self, key, value):
            try:
                self[key]
            except KeyError:
                super(Dictlist, self).__setitem__(key, [])
            self[key].append(value)

    dom_urls = Dictlist()

    pb = ProgressBar(total=len(list), prefix='find duplicates', decimals=2, length=50, fill='█',
                     zfill='-')
    i = 0
    for url in list:
        dom_urls[urlparse(url).netloc] = url
        i += 1
        pb.print_progress_bar(i)

    list = []

    pb = ProgressBar(total=len(dom_urls), prefix='removing duplicates', decimals=2, length=50, fill='█',
                     zfill='-')
    i=0

    for item in dom_urls.values():
        list.append(item[randint(0, len(item)-1)])

        i += 1
        pb.print_progress_bar(i)

    return list


def filter_url_list(url_list):
    return [url for url in url_list if
            not re.search('"|javascript:|void(0)|((\.js|\.png|\.css|\.ico|\.jpg|\.json|\.csv|\.xml|/#)$)|{|}', url) and
            not re.search('(\.js|\.png|\.css|\.ico|\.jpg|\.json|\.csv|\.xml|/#)$', urlparse(url).path)]


def generate_legitimate_urls(N):
    pb = ProgressBar(total=N, prefix='generate legitimate urls', decimals=2, length=50, fill='█',
                     zfill='-')

    domain_list = pandas.read_csv("data/ranked_domains/14-1-2021.csv",
                                  header=None)[1].tolist()

    url_list = []
    search_1 = re.compile('"|javascript:|void(0)|((\.js|\.png|\.css|\.ico|\.jpg|\.json|\.csv|\.xml|/#)$)').search
    search_2 = re.compile('(\.js|\.png|\.css|\.ico|\.jpg|\.json|\.csv|\.xml|/#)$').search

    def url_thread(domain):
        if len(url_list) >= N:
            return

        url = search_for_vulnerable_URLs(domain)

        if url and not search_1(url) and not search_2(urlparse(url).path):
            url_list.append(url)
            pb.print_progress_bar(len(url_list))

            if len(url_list) % 10 == 0:
                pandas.DataFrame(url_list).to_csv(
                    "data/urls/legitimate/{0}.csv".format(date.today().strftime("%d-%m-%Y")), index=False,
                    header=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(url_thread, domain_list)

    url_list = find_duplicates(filter_url_list(url_list))
    pandas.DataFrame(url_list).to_csv("data/urls/legitimate/{0}.csv".format(date.today().strftime("%d-%m-%Y")), index=False, header=False)


def search_for_vulnerable_URLs(domain):
    url = 'http://' + domain

    state, request = is_URL_accessible(url, 1)

    if state:
        r_url = request.url
        content = request.content
        hostname, domain, path = get_domain(r_url)
        extracted_domain = tldextract.extract(r_url)
        domain = extracted_domain.domain + '.' + extracted_domain.suffix

        (Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text), time_extraction = extract_data_from_URL(
            hostname, content,
            domain)
        lst = Href['internals'] + Link['internals']
        lst = ['http://'+lnk for lnk in lst] + Href['externals'] + Link['externals']

        if lst:
            url = lst[randint(1, len(lst))]
            state, url, page = is_URL_accessible(url, 1)
            if state:
                return url

    return None


def extract_features(url, status):
    def words_raw_extraction(domain, subdomain, path):
        w_domain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", domain.lower())
        w_subdomain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", subdomain.lower())
        w_path = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", path.lower())
        raw_words = w_domain + w_path + w_subdomain
        w_host = w_domain + w_subdomain
        return segment(list(filter(None, raw_words))), \
               segment(list(filter(None, w_host))), \
               segment(list(filter(None, w_path)))

    state, request = is_URL_accessible(url)

    if state:
        r_url = request.url
        content = request.content
        hostname, domain, path = get_domain(r_url)
        extracted_domain = tldextract.extract(r_url)
        domain = extracted_domain.domain + '.' + extracted_domain.suffix
        subdomain = extracted_domain.subdomain
        tmp = r_url[r_url.find(extracted_domain.suffix):len(r_url)]
        pth = tmp.partition("/")
        path = pth[1] + pth[2]
        words_raw, words_raw_host, words_raw_path = words_raw_extraction(extracted_domain.domain, subdomain, pth[2])
        tld = extracted_domain.suffix
        parsed = urlparse(r_url)
        scheme = parsed.scheme

        (Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text), time_extraction = extract_data_from_URL(
            hostname, content,
            domain)

        row = [
            [url, time_extraction],

            uf.having_ip_address(url),
            uf.shortening_service(url),

            uf.url_length(r_url),

            uf.count_at(r_url),
            uf.count_exclamation(r_url),
            uf.count_plust(r_url),
            uf.count_sBrackets(r_url),
            uf.count_rBrackets(r_url),
            uf.count_comma(r_url),
            uf.count_dollar(r_url),
            uf.count_semicolumn(r_url),
            uf.count_space(r_url),
            uf.count_and(r_url),
            uf.count_double_slash(r_url),
            uf.count_slash(r_url),
            uf.count_equal(r_url),
            uf.count_percentage(r_url),
            uf.count_question(r_url),
            uf.count_underscore(r_url),
            uf.count_hyphens(r_url),
            uf.count_dots(r_url),
            uf.count_colon(r_url),
            uf.count_star(r_url),
            uf.count_or(r_url),
            uf.count_http_token(r_url),

            uf.https_token(scheme),

            uf.ratio_digits(r_url),
            uf.count_digits(r_url),

            uf.count_tilde(r_url),
            uf.phish_hints(r_url),

            uf.tld_in_path(tld, path),
            uf.tld_in_subdomain(tld, subdomain),
            uf.tld_in_bad_position(tld, subdomain, path),

            uf.abnormal_subdomain(r_url),

            uf.count_redirection(request),
            uf.count_external_redirection(request, domain),

            uf.random_domain(domain),

            uf.random_words(words_raw),
            uf.random_words(words_raw_host),
            uf.random_words(words_raw_path),

            uf.char_repeat(words_raw),
            uf.char_repeat(words_raw_host),
            uf.char_repeat(words_raw_path),

            uf.punycode(r_url),
            uf.domain_in_brand(domain),
            uf.brand_in_path(domain, words_raw_path),
            uf.check_www(words_raw),
            uf.check_com(words_raw),

            uf.port(r_url),

            uf.length_word_raw(words_raw),
            uf.average_word_length(words_raw),
            uf.longest_word_length(words_raw),
            uf.shortest_word_length(words_raw),

            uf.prefix_suffix(r_url),

            uf.count_subdomain(r_url),




            status
        ]

        # print(row)
        return row
    return None


def URLs_analyser(url_list):
    pb = ProgressBar(total=len(url_list), prefix='url analysis', decimals=2, length=50, fill='█',
                     zfill='-')

    data = []
    counters = []

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def url_thread(url):
        res = extract_features(url)

        if res:
            data.append(res[0])
            counters.append(res[1])
            pb.print_progress_bar(len(data))

            if len(data) % 10 == 0:
                pandas.DataFrame(url_list).to_csv("data/datasets/websites.csv", index=False, header=False)

    for chunk in chunks(url_list, 100):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(url_thread, chunk)
            # TODO: USE OPR API

