import requests
import re
from tools import benchmark

session = requests.session()

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

@benchmark(2)
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
#               Getting http-requests data
########################################################################################################################


import time


def fetch(url):
    try:
        return session.get(url, timeout=1)
    except:
        return None


import concurrent.futures


@benchmark(60)
def get_reqs_data(url_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            return_value = [future for future in executor.map(fetch, url_list, timeout=60)]
        except:
            return_value = -5
    return return_value


@benchmark(2)
def count_reqs_error(reqs):
    count = 0

    if len(reqs) > 0:
        for req in reqs:
            if req:
                if req.status_code >= 400:
                    count += 1

        return count / len(reqs)
    else:
        return 0


@benchmark(2)
def count_reqs_redirections(reqs):
    count = 0

    if len(reqs) > 0:
        for req in reqs:
            if req:
                if req.is_redirect:
                    count += 1

        return count / len(reqs)
    else:
        return 0


########################################################################################################################
#               Having login form link
########################################################################################################################

@benchmark(2)
def login_form(Form):
    if len(Form['externals']) > 0 or len(Form['null']) > 0:
        return 1

    p = re.compile('([a-zA-Z0-9\_])+.php')
    for form in Form['internals'] + Form['externals']:
        if p.match(form) != None:
            return 1
    return 0


########################################################################################################################
#               Submitting to email
########################################################################################################################

@benchmark(2)
def submitting_to_email(Form):
    for form in Form['internals'] + Form['externals']:
        if "mailto:" in form or "mail()" in form:
            return 1
    return 0


########################################################################################################################
#               Check for empty title
########################################################################################################################

@benchmark(2)
def empty_title(Title):
    if Title:
        return 0
    return 1


########################################################################################################################
#               ratio of anchor
########################################################################################################################

@benchmark(2)
def ratio_anchor(Anchor, key):
    total = len(Anchor['safe']) + len(Anchor['unsafe'])

    if total == 0:
        return 0
    else:
        return len(Anchor[key]) / total


########################################################################################################################
#              IFrame Redirection
########################################################################################################################

@benchmark(2)
def iframe(IFrame):
    if len(IFrame['invisible']) > 0:
        return 1
    return 0


########################################################################################################################
#              Onmouse action
########################################################################################################################

@benchmark(2)
def onmouseover(content):
    if 'onmouseover="window.status=' in str(content).lower().replace(" ", ""):
        return 1
    else:
        return 0


########################################################################################################################
#              Pop up window
########################################################################################################################

@benchmark(2)
def popup_window(content):
    if "prompt(" in str(content).lower():
        return 1
    else:
        return 0


########################################################################################################################
#              Right_clic action
########################################################################################################################

@benchmark(2)
def right_clic(content):
    if re.findall(r"event.button\s*==\s*2", content):
        return 1
    else:
        return 0


########################################################################################################################
#              Domain in page title/body
########################################################################################################################

@benchmark(2)
def domain_in_text(second_level_domain, text):
    if second_level_domain.lower() in text.lower():
        return 1
    return 0


########################################################################################################################
#              Domain after copyright logo
########################################################################################################################

@benchmark(2)
def domain_with_copyright(domain, content):
    try:
        m = re.search(u'(\N{COPYRIGHT SIGN}|\N{TRADE MARK SIGN}|\N{REGISTERED SIGN})', content)
        _copyright = content[m.span()[0] - 50:m.span()[0] + 50]
        if domain.lower() in _copyright.lower():
            return 1
        else:
            return 0
    except:
        return 0


########################################################################################################################
#              Compression ratio
########################################################################################################################


@benchmark(2)
def compression_ratio(request):
    try:
        compressed_length = int(request.headers['content-length'])
        decompressed_length = len(request.content)
        return compressed_length / decompressed_length
    except:
        return -1


########################################################################################################################
#              Count of text areas
########################################################################################################################

@benchmark(3)
def count_textareas(content):
    soup = BeautifulSoup(content, 'html.parser')

    io_count = len(soup.find_all('textarea')) + len(soup.find_all('input', type=None))
    for io in soup.find_all('input', type=True):
        if io['type'] == 'text' or io['type'] == 'password' or io['type'] == 'search':
            io_count += 1

    return io_count


########################################################################################################################
########################################################################################################################
#                                               JAVASCRIPT
########################################################################################################################
########################################################################################################################


def get_html_from_js(context):
    pattern = r"([\"'`])[\s\w]*(<\s*(\w+)[^>]*>.*<\s*\/\s*\3\s*>)[\s\w]*\1"
    return " ".join([res.group(2) for res in re.finditer(pattern, context, re.MULTILINE) if res.group(2) is not None])


from bs4 import BeautifulSoup
import re


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

@benchmark(2)
def ratio_dynamic_html(s_html, d_html):
    return max(0, min(1, len(d_html) / len(s_html)))


########################################################################################################################
#              Ratio html content on js-code
########################################################################################################################

@benchmark(2)
def ratio_js_on_html(html_context):
    if len(html_context):
        return len(get_html_from_js(remove_JScomments(html_context))) / len(html_context)
    else:
        return 0


########################################################################################################################
#              Amount of http request operations (popular)
########################################################################################################################

@benchmark(3)
def count_io_commands(string):
    pattern = r"(\".*?\"|\'.*?\'|\`.*?\`)|" \
              r"((.(open|send)|$.(get|post|ajax|getJSON)|fetch|axios(|.(get|post|all))|getData)\s*\()"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    count = 0

    for m in re.finditer(regex, string):
        if not m.group(2) and m.groups():
            count += 1

    return count


########################################################################################################################
#                       OCR
########################################################################################################################

import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

langs = '+'.join(pytesseract.get_languages(config=''))


def url_to_image(url):
    resp = session.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def preprocessing_image(img):
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def image_to_text(img):
    if type(img) == list:
        docs = []

        for item in img:
            try:
                txt = pytesseract.image_to_data(url_to_image(img), lang=langs)
                if txt:
                    docs.append(txt)
            except:
                continue

        return ' '.join(docs)

    else:
        try:
            txt = pytesseract.image_to_data(img, lang=langs)
            return txt
        except:
            return ""


########################################################################################################################
#                   Relationship between image text and context
########################################################################################################################


@benchmark(2)
def ratio_Txt(dynamic, static):
    total = len(static)

    if total:
        return min(len(dynamic) / total, 1)
    else:
        return 0


########################################################################################################################
#                   count of phish hints
########################################################################################################################

@benchmark(10)
def count_phish_hints(word_raw, phish_hints):
    if type(word_raw) == list:
        word_raw = ' '.join(word_raw).lower()

    exp = '|'.join([item for sublist in phish_hints.values() for item in sublist])

    if exp:
        return len(re.findall(exp, word_raw))
    else:
        return -1
