import concurrent.futures
from console_progressbar import ProgressBar
from datetime import date
import pandas
import tldextract
import requests
import os
import re
from random import randint
import io
import numpy as np
import time

from data.collector import dir_path
from requests import request

from tools import tokenize, segment, clear_text, compute_tf, compute_idf, compute_tf_idf, benchmark

from urllib.parse import urlparse, urlsplit, urljoin
from bs4 import BeautifulSoup
from collections import Counter

import url_features as uf
import external_features as ef
import content_features as cf

from googletrans import Translator

translator = Translator()

def load_phishHints():
    hints_dir = "data/phish_hints/"
    file_list = os.listdir(hints_dir)

    if file_list:
        return {leng[0:2]: pandas.read_csv(hints_dir + leng, header=None)[0].tolist() for leng in file_list}
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


def check_Language(text):
    global phish_hints

    language = translator.detect(str(text)[0:5000]).lang

    if language not in phish_hints.keys():
        words = translator.translate(" ".join(phish_hints['en'][0:16]), src='en', dest=language).text.split(" ")

        phish_hints[language] = [str(word.strip()) for word in words]

        data = pandas.DataFrame(phish_hints[language])
        filename = "data/phish_hints/{0}.csv".format(language)
        data.to_csv(filename, index=False, header=False)

    return language


key = open("OPR_key.txt").read()


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

@benchmark(20)
def extract_data_from_URL(hostname, content, domain, base_url):
    Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                   "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

    Href = {'internals': [], 'externals': [], 'null': []}
    Link = {'internals': [], 'externals': [], 'null': []}
    Anchor = {'safe': [], 'unsafe': [], 'null': []}
    Img = {'internals': [], 'externals': [], 'null': []}
    Media = {'internals': [], 'externals': [], 'null': []}
    Form = {'internals': [], 'externals': [], 'null': []}
    CSS = {'internals': [], 'externals': [], 'null': [], 'embedded': 0}
    Favicon = {'internals': [], 'externals': [], 'null': []}
    IFrame = {'visible': [], 'invisible': [], 'null': []}
    SCRIPT = {'internals': [], 'externals': [], 'null': [], 'embedded': 0}       # JavaScript
    Title = ''
    Text = ''

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

    # collect all css
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

    # collect all form actions
    for form in soup.findAll('form', action=True):
        url = form['action']

        if url in Null_format or url == 'about:blank':
            Form['null'].append('http://' + hostname + '/' + url)
            continue

        url = urljoin(base_url, url)

        if domain in urlparse(url).netloc:
            Form['internals'].append(url)
        else:
            Form['externals'].append(url)

    # collect all link tags
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
        Title = soup.title.string.lower()
    except:
        pass

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

    internals_script_doc = ' '.join(
        [internals_script_doc] + [script.contents[0] for script in soup.find_all('script', src=False) if
                                  len(script.contents) > 0])
    SCRIPT['embedded'] = len([script.contents for script in soup.find_all('script', src=False) if len(script.contents) > 0])

    return Href, Link, Anchor, Media, Img, Form, CSS, Favicon, IFrame, SCRIPT, Title, Text, internals_script_doc, externals_script_doc


filter1 = re.compile('"|javascript:|void(0)|((\.js|\.png|\.css|\.ico|\.jpg|\.json|\.csv|\.xml|/#)$)').search
filter2 = re.compile('(\.js|\.png|\.css|\.ico|\.jpg|\.json|\.csv|\.xml|/#)$').search


def extract_URLs_from_page(hostname, content, domain):
    Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                   "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

    Href = []
    soup = BeautifulSoup(content, 'html.parser')

    # collect all external and internal hrefs from url
    for href in soup.find_all('a', href=True):
        dots = [x.start(0) for x in re.finditer('\.', href['href'])]
        if hostname in href['href'] or domain in href['href'] or len(dots) == 1 or not href['href'].startswith(
                'http'):
            if not href['href'].startswith('http'):
                if not href['href'].startswith('/'):
                    Href.append('http://' + hostname + '/' + href['href'])
                elif href['href'] not in Null_format:
                    Href.append('http://' + hostname + href['href'])
        else:
            Href.append(href['href'])

    Href = [url for url in Href if url if url and not filter1(url) and not filter2(urlparse(url).path)]
    # Href = [url for url in Href if is_URL_accessible(url)[0]]

    return Href


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
    i = 0

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

    def url_search(domain):
        if len(url_list) >= N:
            return

        url = search_for_vulnerable_URLs(domain)

        if url:
            url_list.append(url)
            pb.print_progress_bar(len(url_list))

            if len(url_list) % 10 == 0:
                pandas.DataFrame(url_list).to_csv(
                    "data/urls/legitimate/{0}.csv".format(date.today().strftime("%d-%m-%Y")), index=False,
                    header=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(url_search, domain_list)

    url_list = find_duplicates(filter_url_list(url_list))
    pandas.DataFrame(url_list).to_csv("data/urls/legitimate/{0}.csv".format(date.today().strftime("%d-%m-%Y")),
                                      index=False, header=False)


def search_for_vulnerable_URLs(domain):
    url = 'http://' + domain

    state, request = is_URL_accessible(url, 1)

    if state:
        url = request.url
        content = request.content
        hostname, domain, path = get_domain(url)
        extracted_domain = tldextract.extract(url)
        domain = extracted_domain.domain + '.' + extracted_domain.suffix

        Href = extract_URLs_from_page(hostname, content, domain)

        if Href:
            url = Href[randint(0, len(Href))-1]
            state, request = is_URL_accessible(url, 1)
            if state:
                return request.url

    return None

headers = {
    'stats': [
        'word_ratio',

                    #   URL FEATURES

                    'uf.having_ip_address(url)',
                    'uf.shortening_service(url)',

                    "cert!=None",

                    'uf.url_length(r_url)',

                    'uf.count_at(r_url)',
                    'uf.count_exclamation(r_url)',
                    'uf.count_plust(r_url)',
                    'uf.count_sBrackets(r_url)',
                    'uf.count_rBrackets(r_url)',
                    'uf.count_comma(r_url)',
                    'uf.count_dollar(r_url)',
                    'uf.count_semicolumn(r_url)',
                    'uf.count_space(r_url)',
                    'uf.count_and(r_url)',
                    'uf.count_double_slash(r_url)',
                    'uf.count_slash(r_url)',
                    'uf.count_equal(r_url)',
                    'uf.count_percentage(r_url)',
                    'uf.count_question(r_url)',
                    'uf.count_underscore(r_url)',
                    'uf.count_hyphens(r_url)',
                    'uf.count_dots(r_url)',
                    'uf.count_colon(r_url)',
                    'uf.count_star(r_url)',
                    'uf.count_or(r_url)',
                    'uf.count_tilde(r_url)',
                    'uf.count_http_token(r_url)',

                    'uf.https_token(scheme)',

                    'uf.ratio_digits(r_url)',
                    'uf.count_digits(r_url)',

                    'cf.count_phish_hints(r_url,phish_hints)',

                    'uf.tld_in_path(tld,path)',
                    'uf.tld_in_subdomain(tld,subdomain)',
                    'uf.tld_in_bad_position(tld,subdomain,path)',
                    'uf.abnormal_subdomain(r_url)',

                    'uf.count_redirection(request)',
                    'uf.count_external_redirection(request,domain)',

                    'uf.random_domain(second_level_domain)',

                    'uf.random_words(words_raw)',
                    'uf.random_words(words_raw_host)',
                    'uf.random_words(words_raw_path)',

                    'uf.char_repeat(words_raw)',
                    'uf.char_repeat(words_raw_host)',
                    'uf.char_repeat(words_raw_path)',

                    'uf.punycode(r_url)',
                    'uf.domain_in_brand(second_level_domain)',
                    'uf.brand_in_path(second_level_domain, words_raw_path)',
                    'uf.check_www(words_raw)',
                    'uf.check_com(words_raw)',

                    'uf.port(r_url)',

                    'uf.length_word_raw(words_raw)',
                    'uf.average_word_length(words_raw)',
                    'uf.longest_word_length(words_raw)',
                    'uf.shortest_word_length(words_raw)',

                    'uf.prefix_suffix(r_url)',

                    'uf.count_subdomain(r_url)',

                    'uf.count_visual_similarity_domains(second_level_domain)',

                    #   CONTENT FEATURE
                    #       (static)

                    'cf.compression_ratio(request)',
                    'cf.count_textareas(content)',
                    'cf.ratio_js_on_html(Text)',

                    'len(iUrl_s)+len(eUrl_s)',
                    'cf.urls_ratio(iUrl_s,iUrl_s+eUrl_s+nUrl_s)',
                    'cf.urls_ratio(eUrl_s,iUrl_s+eUrl_s+nUrl_s)',
                    'cf.urls_ratio(nUrl_s,iUrl_s+eUrl_s+nUrl_s)',
                    "cf.ratio_List(CSS,'internals')",
                    "cf.ratio_List(CSS,'externals')",
                    "cf.ratio_List(CSS,'embedded')",
                    "cf.ratio_List(SCRIPT,'internals')",
                    "cf.ratio_List(SCRIPT,'externals')",
                    "cf.ratio_List(SCRIPT,'embedded')",
                    "cf.ratio_List(Img,'externals')",
                    "cf.ratio_List(Img,'internals')",
                    "cf.count_reqs_redirections(reqs_iData_s)",
                    "cf.count_reqs_redirections(reqs_eData_s)",
                    "cf.count_reqs_error(reqs_iData_s)",
                    "cf.count_reqs_error(reqs_eData_s)",
                    "cf.login_form(Form)",
                    "cf.ratio_List(Favicon,'externals')",
                    "cf.ratio_List(Favicon,'internals')",
                    "cf.submitting_to_email(Form)",
                    "cf.ratio_List(Media,'internals')",
                    "cf.ratio_List(Media,'externals')",
                    "cf.empty_title(Title)",
                    "cf.ratio_anchor(Anchor,'unsafe')",
                    "cf.ratio_anchor(Anchor,'safe')",
                    "cf.ratio_List(Link,'internals')",
                    "cf.ratio_List(Link,'externals')",
                    "cf.iframe(IFrame)",
                    "cf.onmouseover(content)",
                    "cf.popup_window(content)",
                    "cf.right_clic(content)",
                    "cf.domain_in_text(second_level_domain,Text)",
                    "cf.domain_in_text(second_level_domain,Title)",
                    "cf.domain_with_copyright(domain,content)",
                    "cf.count_phish_hints(Text,phish_hints)",
                    "cf.ratio_Txt(iImgTxt_words+eImgTxt_words,sContent_words)",
                    "cf.ratio_Txt(iImgTxt_words,sContent_words)",
                    "cf.ratio_Txt(eImgTxt_words,sContent_words)",
                    "cf.ratio_Txt(eImgTxt_words,iImgTxt_words)",

                    #       (dynamic)

                    "cf.ratio_dynamic_html(Text,"".join([Text_di,Text_de]))",

                    #       (dynamic internals)

                    "cf.ratio_dynamic_html(Text,Text_di)",
                    "cf.ratio_js_on_html(Text_di)",
                    "cf.count_textareas(content_di)",

                    "len(iUrl_di)+len(eUrl_di)",
                    "cf.urls_ratio(iUrl_di,iUrl_di+eUrl_di+nUrl_di)",
                    "cf.urls_ratio(eUrl_di,iUrl_di+eUrl_di+nUrl_di)",
                    "cf.urls_ratio(nUrl_di,iUrl_di+eUrl_di+nUrl_di)",
                    "cf.ratio_List(CSS_di,'internals')",
                    "cf.ratio_List(CSS_di,'externals')",
                    "cf.ratio_List(CSS_di,'embedded')",
                    "cf.ratio_List(SCRIPT_di,'internals')",
                    "cf.ratio_List(SCRIPT_di,'externals')",
                    "cf.ratio_List(SCRIPT_di,'embedded')",
                    "cf.ratio_List(Img_di,'externals')",
                    "cf.ratio_List(Img_di,'internals')",
                    "cf.count_reqs_redirections(reqs_iData_di)",
                    "cf.count_reqs_redirections(reqs_eData_di)",
                    "cf.count_reqs_error(reqs_iData_di)",
                    "cf.count_reqs_error(reqs_eData_di)",
                    "cf.login_form(Form_di)",
                    "cf.ratio_List(Favicon_di,'externals')",
                    "cf.ratio_List(Favicon_di,'internals')",
                    "cf.submitting_to_email(Form_di)",
                    "cf.ratio_List(Media_di,'internals')",
                    "cf.ratio_List(Media_di,'externals')",
                    "cf.empty_title(Title_di)",
                    "cf.ratio_anchor(Anchor_di,'unsafe')",
                    "cf.ratio_anchor(Anchor_di,'safe')",
                    "cf.ratio_List(Link_di,'internals')",
                    "cf.ratio_List(Link_di,'externals')",
                    "cf.iframe(IFrame_di)",
                    "cf.onmouseover(content_di)",
                    "cf.popup_window(content_di)",
                    "cf.right_clic(content_di)",
                    "cf.domain_in_text(second_level_domain,Text_di)",
                    "cf.domain_in_text(second_level_domain,Title_di)",
                    "cf.domain_with_copyright(domain,content_di)",

                    "cf.count_io_commands(internals_script_doc)",
                    "cf.count_phish_hints(Text_di,phish_hints)",

                    #       (dynamic externals)

                    "cf.ratio_dynamic_html(Text,Text_de)",
                    "cf.ratio_js_on_html(Text_de)",
                    "cf.count_textareas(content_de)",

                    "len(iUrl_de)+len(eUrl_de)",
                    "cf.urls_ratio(iUrl_de,iUrl_de+eUrl_de+nUrl_de)",
                    "cf.urls_ratio(eUrl_de,iUrl_de+eUrl_de+nUrl_de)",
                    "cf.urls_ratio(nUrl_de,iUrl_de+eUrl_de+nUrl_de)",
                    "cf.ratio_List(CSS_de,'internals')",
                    "cf.ratio_List(CSS_de,'externals')",
                    "cf.ratio_List(CSS_de,'embedded')",
                    "cf.ratio_List(SCRIPT_de,'internals')",
                    "cf.ratio_List(SCRIPT_de,'externals')",
                    "cf.ratio_List(SCRIPT_de,'embedded')",
                    "cf.ratio_List(Img_de,'externals')",
                    "cf.ratio_List(Img_de,'internals')",
                    "cf.count_reqs_redirections(reqs_iData_de)",
                    "cf.count_reqs_redirections(reqs_eData_de)",
                    "cf.count_reqs_error(reqs_iData_de)",
                    "cf.count_reqs_error(reqs_eData_de)",
                    "cf.login_form(Form_de)",
                    "cf.ratio_List(Favicon_de,'externals')",
                    "cf.ratio_List(Favicon_de,'internals')",
                    "cf.submitting_to_email(Form_de)",
                    "cf.ratio_List(Media_de,'internals')",
                    "cf.ratio_List(Media_de,'externals')",
                    "cf.empty_title(Title_de)",
                    "cf.ratio_anchor(Anchor_de,'unsafe')",
                    "cf.ratio_anchor(Anchor_de,'safe')",
                    "cf.ratio_List(Link_de,'internals')",
                    "cf.ratio_List(Link_de,'externals')",
                    "cf.iframe(IFrame_de)",
                    "cf.onmouseover(content_de)",
                    'cf.popup_window(content_de)',
                    "cf.right_clic(content_de)",
                    "cf.domain_in_text(second_level_domain,Text_de)",
                    "cf.domain_in_text(second_level_domain,Title_de)",
                    "cf.domain_with_copyright(domain,content_de)",

                    "cf.count_io_commands(externals_script_doc)",
                    "cf.count_phish_hints(Text_de,phish_hints)",

                    #   EXTERNAL FEATURES

                    'ef.domain_registration_length(domain)',
                    "ef.whois_registered_domain(domain)",
                    "ef.web_traffic(r_url)",
                    # "ef.google_index(r_url)",
                    "ef.page_rank(domain)",
                    # "ef.compare_search2url(r_url,domain,TF.most_common(5)])",
                    "ef.remainder_valid_cert(hostinfo.cert)",
                    "ef.valid_cert_period(hostinfo.cert)",
                    "ef.count_alt_names(hostinfo.cert)"
    ],
    'metadata': [
        'url',
        'lang',
        'status'
    ],
    'substats': [
        'extraction-contextData-time',
        'image-recognition-time'
    ]
}

@benchmark(200)
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
        content = str(request.content)
        hostname, second_level_domain, path = get_domain(r_url)
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

        cert, time_cert = ef.get_cert(domain)

        (Href, Link, Anchor, Media, Img, Form, CSS, Favicon, IFrame, SCRIPT, Title, Text, internals_script_doc,
         externals_script_doc), extraction_data_time = extract_data_from_URL(hostname, content, domain, r_url)

        content_di = cf.get_html_from_js(cf.remove_JScomments(internals_script_doc))
        content_de = cf.get_html_from_js(cf.remove_JScomments(externals_script_doc))

        (Href_di, Link_di, Anchor_di, Media_di, Img_di, Form_di, CSS_di, Favicon_di, IFrame_di, SCRIPT_di, Title_di,
         Text_di, internals_script_doc_di, externals_script_doc_di), extraction_data_di_time = extract_data_from_URL(
            hostname, content_di, domain, r_url)

        (Href_de, Link_de, Anchor_de, Media_de, Img_de, Form_de, CSS_de, Favicon_de, IFrame_de, SCRIPT_de, Title_de,
         Text_de, internals_script_doc_de, externals_script_doc_de), extraction_data_de_time = extract_data_from_URL(
            hostname, content_de, domain, r_url)


        start = time.time()
        internals_img_txt = cf.image_to_text(Img['internals'])
        externals_img_txt = cf.image_to_text(Img['externals'])
        extracting_ImgTxt_time = time.time() - start


        start = time.time()
        iImgTxt_words = clear_text(tokenize(internals_img_txt.lower()))
        eImgTxt_words = clear_text(tokenize(externals_img_txt.lower()))

        sContent_words = clear_text(tokenize(Text.lower()))
        diContent_words = clear_text(tokenize(Text_di.lower()))
        deContent_words = clear_text(tokenize(Text_de.lower()))

        Text_words = iImgTxt_words + eImgTxt_words + sContent_words + diContent_words + deContent_words
        TF = compute_tf(Text_words)
        word_ratio = len(TF) / len(Text_words)
        preparing_words_time = time.time() - start

        lang = check_Language(content)

        iUrl_s = Href['internals'] + Link['internals'] + Media['internals'] + Form['internals']
        eUrl_s = Href['externals'] + Link['externals'] + Media['externals'] + Form['externals']
        nUrl_s = Href['null'] + Link['null'] + Media['null'] + Form['null']

        reqs_iData_s, reqs_iTime_s = cf.get_reqs_data(iUrl_s)
        reqs_eData_s, reqs_eTime_s = cf.get_reqs_data(eUrl_s)


        iUrl_di = Href_di['internals'] + Link_di['internals'] + Media_di['internals'] + Form_di['internals']
        eUrl_di = Href_di['externals'] + Link_di['externals'] + Media_di['externals'] + Form_di['externals']
        nUrl_di = Href_di['null'] + Link_di['null'] + Media_di['null'] + Form_di['null']

        reqs_iData_di, reqs_iTime_di = cf.get_reqs_data(iUrl_di)
        reqs_eData_di, reqs_eTime_di = cf.get_reqs_data(eUrl_di)


        iUrl_de = Href_de['internals'] + Link_de['internals'] + Media_de['internals'] + Form_de['internals']
        eUrl_de = Href_de['externals'] + Link_de['externals'] + Media_de['externals'] + Form_de['externals']
        nUrl_de = Href_de['null'] + Link_de['null'] + Media_de['null'] + Form_de['null']

        reqs_iData_de, reqs_iTime_de = cf.get_reqs_data(iUrl_de)
        reqs_eData_de, reqs_eTime_de = cf.get_reqs_data(eUrl_de)

        record = {
            'url': url,
            'domain': domain,
            'lang': lang,
            'TF': TF,
            'status': status,
            'extraction-contextData-time': extraction_data_time + extraction_data_di_time + extraction_data_de_time,
            'image-recognition-time': extracting_ImgTxt_time,
            'stats':
                [
                    (word_ratio, preparing_words_time),

                    #   URL FEATURES

                    uf.having_ip_address(url),
                    uf.shortening_service(url),

                    (cert != None, time_cert),

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
                    uf.count_tilde(r_url),
                    uf.count_http_token(r_url),

                    uf.https_token(scheme),

                    uf.ratio_digits(r_url),
                    uf.count_digits(r_url),

                    cf.count_phish_hints(r_url, phish_hints),

                    uf.tld_in_path(tld, path),
                    uf.tld_in_subdomain(tld, subdomain),
                    uf.tld_in_bad_position(tld, subdomain, path),
                    uf.abnormal_subdomain(r_url),

                    uf.count_redirection(request),
                    uf.count_external_redirection(request, domain),

                    uf.random_domain(second_level_domain),

                    uf.random_words(words_raw),
                    uf.random_words(words_raw_host),
                    uf.random_words(words_raw_path),

                    uf.char_repeat(words_raw),
                    uf.char_repeat(words_raw_host),
                    uf.char_repeat(words_raw_path),

                    uf.punycode(r_url),
                    uf.domain_in_brand(second_level_domain),
                    uf.brand_in_path(second_level_domain, words_raw_path),
                    uf.check_www(words_raw),
                    uf.check_com(words_raw),

                    uf.port(r_url),

                    uf.length_word_raw(words_raw),
                    uf.average_word_length(words_raw),
                    uf.longest_word_length(words_raw),
                    uf.shortest_word_length(words_raw),

                    uf.prefix_suffix(r_url),

                    uf.count_subdomain(r_url),

                    uf.count_visual_similarity_domains(second_level_domain),

                    #   CONTENT FEATURE
                    #       (static)

                    cf.compression_ratio(request),
                    cf.count_textareas(content),
                    cf.ratio_js_on_html(Text),

                    (len(iUrl_s) + len(eUrl_s), reqs_iTime_s + reqs_eTime_s),
                    (cf.urls_ratio(iUrl_s, iUrl_s + eUrl_s + nUrl_s), reqs_iTime_s),
                    (cf.urls_ratio(eUrl_s, iUrl_s + eUrl_s + nUrl_s), reqs_eTime_s),
                    (cf.urls_ratio(nUrl_s, iUrl_s + eUrl_s + nUrl_s), 0),
                    cf.ratio_List(CSS, 'internals'),
                    cf.ratio_List(CSS, 'externals'),
                    cf.ratio_List(CSS, 'embedded'),
                    cf.ratio_List(SCRIPT, 'internals'),
                    cf.ratio_List(SCRIPT, 'externals'),
                    cf.ratio_List(SCRIPT, 'embedded'),
                    cf.ratio_List(Img, 'externals'),
                    cf.ratio_List(Img, 'internals'),
                    cf.count_reqs_redirections(reqs_iData_s),
                    cf.count_reqs_redirections(reqs_eData_s),
                    cf.count_reqs_error(reqs_iData_s),
                    cf.count_reqs_error(reqs_eData_s),
                    cf.login_form(Form),
                    cf.ratio_List(Favicon, 'externals'),
                    cf.ratio_List(Favicon, 'internals'),
                    cf.submitting_to_email(Form),
                    cf.ratio_List(Media, 'internals'),
                    cf.ratio_List(Media, 'externals'),
                    cf.empty_title(Title),
                    cf.ratio_anchor(Anchor, 'unsafe'),
                    cf.ratio_anchor(Anchor, 'safe'),
                    cf.ratio_List(Link, 'internals'),
                    cf.ratio_List(Link, 'externals'),
                    cf.iframe(IFrame),
                    cf.onmouseover(content),
                    cf.popup_window(content),
                    cf.right_clic(content),
                    cf.domain_in_text(second_level_domain, Text),
                    cf.domain_in_text(second_level_domain, Title),
                    cf.domain_with_copyright(domain, content),
                    cf.count_phish_hints(Text, phish_hints),
                    cf.ratio_Txt(iImgTxt_words + eImgTxt_words, sContent_words),
                    cf.ratio_Txt(iImgTxt_words, sContent_words),
                    cf.ratio_Txt(eImgTxt_words, sContent_words),
                    cf.ratio_Txt(eImgTxt_words, iImgTxt_words),

                    #       (dynamic)

                    cf.ratio_dynamic_html(Text, "".join([Text_di, Text_de])),

                    #       (dynamic internals)

                    cf.ratio_dynamic_html(Text, Text_di),
                    cf.ratio_js_on_html(Text_di),
                    cf.count_textareas(content_di),

                    (len(iUrl_di) + len(eUrl_di), reqs_iTime_de + reqs_eTime_de),
                    (cf.urls_ratio(iUrl_di, iUrl_di + eUrl_di + nUrl_di + nUrl_di), reqs_iTime_di),
                    (cf.urls_ratio(eUrl_di, iUrl_di + eUrl_di + nUrl_di), reqs_eTime_di),
                    (cf.urls_ratio(nUrl_di, iUrl_di + eUrl_di + nUrl_di), 0),
                    cf.ratio_List(CSS_di, 'internals'),
                    cf.ratio_List(CSS_di, 'externals'),
                    cf.ratio_List(CSS_di, 'embedded'),
                    cf.ratio_List(SCRIPT_di, 'internals'),
                    cf.ratio_List(SCRIPT_di, 'externals'),
                    cf.ratio_List(SCRIPT_di, 'embedded'),
                    cf.ratio_List(Img_di, 'externals'),
                    cf.ratio_List(Img_di, 'internals'),
                    cf.count_reqs_redirections(reqs_iData_di),
                    cf.count_reqs_redirections(reqs_iData_di),
                    cf.count_reqs_error(reqs_iData_di),
                    cf.count_reqs_error(reqs_iData_di),
                    cf.login_form(Form_di),
                    cf.ratio_List(Favicon_di, 'externals'),
                    cf.ratio_List(Favicon_di, 'internals'),
                    cf.submitting_to_email(Form_di),
                    cf.ratio_List(Media_di, 'internals'),
                    cf.ratio_List(Media_di, 'externals'),
                    cf.empty_title(Title_di),
                    cf.ratio_anchor(Anchor_di, 'unsafe'),
                    cf.ratio_anchor(Anchor_di, 'safe'),
                    cf.ratio_List(Link_di, 'internals'),
                    cf.ratio_List(Link_di, 'externals'),
                    cf.iframe(IFrame_di),
                    cf.onmouseover(content_di),
                    cf.popup_window(content_di),
                    cf.right_clic(content_di),
                    cf.domain_in_text(second_level_domain, Text_di),
                    cf.domain_in_text(second_level_domain, Title_di),
                    cf.domain_with_copyright(domain, content_di),

                    cf.count_io_commands(internals_script_doc),
                    cf.count_phish_hints(Text_di, phish_hints),

                    #       (dynamic externals)

                    cf.ratio_dynamic_html(Text, Text_de),
                    cf.ratio_js_on_html(Text_de),
                    cf.count_textareas(content_de),

                    (len(iUrl_de) + len(eUrl_de), reqs_iTime_de + reqs_eTime_de),
                    (cf.urls_ratio(iUrl_de, iUrl_de + eUrl_de + nUrl_de), reqs_iTime_de),
                    (cf.urls_ratio(eUrl_de, iUrl_de + eUrl_de + nUrl_de), reqs_eTime_de),
                    (cf.urls_ratio(nUrl_de, iUrl_de + eUrl_de + nUrl_de), 0),
                    cf.ratio_List(CSS_de, 'internals'),
                    cf.ratio_List(CSS_de, 'externals'),
                    cf.ratio_List(CSS_de, 'embedded'),
                    cf.ratio_List(SCRIPT_de, 'internals'),
                    cf.ratio_List(SCRIPT_de, 'externals'),
                    cf.ratio_List(SCRIPT_de, 'embedded'),
                    cf.ratio_List(Img_de, 'externals'),
                    cf.ratio_List(Img_de, 'internals'),
                    cf.count_reqs_redirections(reqs_iData_de),
                    cf.count_reqs_redirections(reqs_eData_de),
                    cf.count_reqs_error(reqs_iData_de),
                    cf.count_reqs_error(reqs_eData_de),
                    cf.login_form(Form_de),
                    cf.ratio_List(Favicon_de, 'externals'),
                    cf.ratio_List(Favicon_de, 'internals'),
                    cf.submitting_to_email(Form_de),
                    cf.ratio_List(Media_de, 'internals'),
                    cf.ratio_List(Media_de, 'externals'),
                    cf.empty_title(Title_de),
                    cf.ratio_anchor(Anchor_de, 'unsafe'),
                    cf.ratio_anchor(Anchor_de, 'safe'),
                    cf.ratio_List(Link_de, 'internals'),
                    cf.ratio_List(Link_de, 'externals'),
                    cf.iframe(IFrame_de),
                    cf.onmouseover(content_de),
                    cf.popup_window(content_de),
                    cf.right_clic(content_de),
                    cf.domain_in_text(second_level_domain, Text_de),
                    cf.domain_in_text(second_level_domain, Title_de),
                    cf.domain_with_copyright(domain, content_de),

                    cf.count_io_commands(externals_script_doc),
                    cf.count_phish_hints(Text_de, phish_hints),

                    #   EXTERNAL FEATURES

                    ef.domain_registration_length(domain),  # do not use VPN: Error trying to connect to socket: closing socket
                    ef.whois_registered_domain(domain),
                    ef.web_traffic(r_url),
                    # ef.google_index(r_url),
                    ef.page_rank(domain),
                    # ef.compare_search2url(r_url, domain, [t[0] for t in Counter(TF).most_common(5)]),
                    ef.remainder_valid_cert(cert),
                    ef.valid_cert_period(cert),
                    ef.count_alt_names(cert)
                ]
        }

        return record
    return None


chunk_size = 10


def generate_dataset(url_list):
    data = []
    counter = []

    TF = {0: [], 1: []}

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    pb_i = []

    pb = ProgressBar(total=len(url_list), prefix='url analysis', decimals=3, length=50, fill='█',
                     zfill='-')
    pb.print_progress_bar(0)

    def extraction_data(obj):
        res, t = extract_features(obj[0], obj[1])
        pb_i.append(t)
        pb.print_progress_bar(len(pb_i))

        if type(res) is dict:
            return res
        else:
            return None


    for chunk in chunks(url_list, chunk_size):
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            buffer = [executor.submit(extraction_data, obj) for obj in chunk]
            concurrent.futures.wait(buffer)

            # buffer = executor.map(extraction_data, chunk)

        res = [r.result() for r in buffer if r.result()]
        # res = [r for r in buffer if r]

        if res:
            tmp = np.array([record['stats'] for record in res])
            metadata = np.array([[record.get(key) for key in headers['metadata']] for record in res])
            substats = np.array([[record.get(key) for key in headers['substats']] for record in res])

            for record in res:
                TF[record['status']].append(record['TF'])

            data += np.c_[metadata, tmp[:, :, 0]].tolist()
            counter += np.c_[substats, tmp[:, :, 1]].tolist()

            pandas.DataFrame(counter).to_csv(dir_path+'feature_times.csv',
                index=False, header=headers['substats'] + headers['stats'])

            pandas.DataFrame(data).to_csv(dir_path+'dataset.csv',
                                          index=False, header=headers['metadata'] + headers['stats'])
            pandas.DataFrame([['{}={}'.format(k, tf_list[k]) for k in tf_list] for tf_list in TF[0]]).to_csv(
                dir_path+'TF (legitimate).csv', index=False, header=False)
            pandas.DataFrame([['{}={}'.format(k, tf_list[k]) for k in tf_list] for tf_list in TF[1]]).to_csv(
                dir_path+'TF (phishing).csv', index=False, header=False)

            IDF = {0: compute_idf(TF[0]), 1: compute_idf(TF[1])}
            TF_IDF = {0: [], 1: []}

            for s, tf_list in TF.items():
                for tf in tf_list:
                    TF_IDF[s].append(compute_tf_idf(tf, IDF[s]))

            pandas.DataFrame(TF_IDF[0]).to_csv(dir_path+'TF-IDF (legitimate).csv', index=False, header=False)
            pandas.DataFrame(TF_IDF[1]).to_csv(dir_path+'TF-IDF (phishing).csv', index=False, header=False)

            t = sum(pb_i) / len(pb_i)
            all_t = t * len(url_list) / len(pb_i)
            print('updated: +{}\t[time left: {} min]'.format(len(res), (all_t - t)/60))