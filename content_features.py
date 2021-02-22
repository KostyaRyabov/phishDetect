from main import state

if state in range(1, 7):
    import requests
    from tools import benchmark

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

    @benchmark(5)
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
            return requests.get(url, timeout=2)
        except:
            return None


    import concurrent.futures
    import configparser

    config = configparser.ConfigParser()
    config.read('settings.ini')
    http_requests_thread_count = int(config['THREADS']['http_requests_thread_count'])
    image_recognition_group_thread_count = int(config['THREADS']['image_recognition_group_thread_count'])

    @benchmark(40)
    def get_reqs_data(url_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=http_requests_thread_count) as executor:
            return_value = [req for req in executor.map(fetch, url_list, timeout=30)]
        return return_value


    @benchmark(5)
    def count_reqs_error(reqs):
        if type(reqs) != list:
            return 0

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


    @benchmark(5)
    def count_reqs_redirections(reqs):
        if type(reqs) != list:
            return 0

        if len(reqs) > 0:
            count = 0
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

    @benchmark(5)
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

    @benchmark(5)
    def submitting_to_email(Form):
        for form in Form['internals'] + Form['externals']:
            if "mailto:" in form or "mail()" in form:
                return 1
        return 0


    ########################################################################################################################
    #               Check for empty title
    ########################################################################################################################

    @benchmark(5)
    def empty_title(Title):
        if Title:
            return 0
        return 1


    ########################################################################################################################
    #               ratio of anchor
    ########################################################################################################################

    @benchmark(5)
    def ratio_anchor(Anchor, key):
        total = len(Anchor['safe']) + len(Anchor['unsafe'])

        if total == 0:
            return 0
        else:
            return len(Anchor[key]) / total


    ########################################################################################################################
    #              IFrame Redirection
    ########################################################################################################################

    @benchmark(5)
    def iframe(IFrame):
        if len(IFrame['invisible']) > 0:
            return 1
        return 0


    ########################################################################################################################
    #              Onmouse action
    ########################################################################################################################

    @benchmark(5)
    def onmouseover(content):
        if 'onmouseover="window.status=' in str(content).lower().replace(" ", ""):
            return 1
        else:
            return 0


    ########################################################################################################################
    #              Pop up window
    ########################################################################################################################

    @benchmark(5)
    def popup_window(content):
        if "prompt(" in str(content).lower():
            return 1
        else:
            return 0


    ########################################################################################################################
    #              Right_clic action
    ########################################################################################################################

    @benchmark(5)
    def right_clic(content):
        if re.findall(r"event.button\s*==\s*2", content):
            return 1
        else:
            return 0


    ########################################################################################################################
    #              Domain in page title/body
    ########################################################################################################################

    @benchmark(5)
    def domain_in_text(second_level_domain, text):
        if second_level_domain.lower() in text.lower():
            return 1
        return 0


    ########################################################################################################################
    #              Domain after copyright logo
    ########################################################################################################################

    @benchmark(5)
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


    @benchmark(5)
    def compression_ratio(request):
        try:
            compressed_length = int(request.headers['content-length'])
            decompressed_length = len(request.content)
            return min(compressed_length / decompressed_length, 1)
        except:
            return 1


    ########################################################################################################################
    #              Count of text areas
    ########################################################################################################################

    @benchmark(5)
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
        # pattern = r"([\"'`])[\s\w]*(<\s*(\w+)[^>]*>.*<\s*\/\s*\3\s*>)[\s\w]*\1"
        pattern = r"([\"'`])[\s\w]*(<\s*(\w+)[^>]*>.*(<\s*\/\s*\3\s*>)?)[\s\w]*\1"
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

    @benchmark(5)
    def ratio_dynamic_html(s_html, d_html):
        return max(0, min(1, len(d_html) / len(s_html)))


    ########################################################################################################################
    #              Ratio html content on js-code
    ########################################################################################################################

    @benchmark(5)
    def ratio_js_on_html(html_context):
        if len(html_context):
            return len(get_html_from_js(remove_JScomments(html_context))) / len(html_context)
        else:
            return 0


    ########################################################################################################################
    #              Amount of http request operations (popular)
    ########################################################################################################################

    @benchmark(5)
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


    from iso639 import languages

    def image_to_text(img, lang):
        try:
            lang = languages.get(alpha2=lang).bibliographic

            if 'eng' not in lang:
                lang = 'eng+' + lang

            if type(img) == list:
                with concurrent.futures.ThreadPoolExecutor(max_workers=image_recognition_group_thread_count) as executor:
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


    @benchmark(5)
    def ratio_Txt(dynamic, static):
        total = len(static)

        if total:
            return min(len(dynamic) / total, 1)
        else:
            return 0


    ########################################################################################################################
    #                   count of phish hints
    ########################################################################################################################

    @benchmark(20)
    def count_phish_hints(word_raw, phish_hints, lang):
        if type(word_raw) == list:
            word_raw = ' '.join(word_raw).lower()

        try:
            exp = '|'.join(list(set([item for sublist in [phish_hints[lang],phish_hints['en']] for item in sublist])))

            if exp:
                return len(re.findall(exp, word_raw))
            else:
                return 0
        except:
            return 0
