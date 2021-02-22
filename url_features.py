from main import state

if state in range(1, 7):

    import re
    from tools import benchmark, segment, WORDS
    from data.collector import brand_list


    def tokenize_url(word_raw):
        return segment(' '.join(word_raw))


    ########################################################################################################################
    #               Having IP address in hostname
    ########################################################################################################################

    @benchmark(5)
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


    ########################################################################################################################
    #               URL hostname length
    ########################################################################################################################

    @benchmark(5)
    def url_length(url):
        return len(url)


    ########################################################################################################################
    #               URL shortening
    ########################################################################################################################

    @benchmark(5)
    def shortening_service(full_url):
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
                          full_url)
        if match:
            return 1
        else:
            return 0


    ########################################################################################################################
    #               Count at ('@') symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_at(base_url):
        return base_url.count('@')


    ########################################################################################################################
    #               Count exclamation mark ('!') symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_exclamation(base_url):
        return base_url.count('!')


    ########################################################################################################################
    #               Count plus ('+') symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_plust(base_url):
        return base_url.count('+')


    ########################################################################################################################
    #               Count square brackets ( '[', ']' ) symbols at base url
    ########################################################################################################################

    @benchmark(5)
    def count_sBrackets(base_url):
        return len(re.findall('[\[\]]', base_url))


    ########################################################################################################################
    #               Count round brackets ( '(', ')' ) symbols at base url
    ########################################################################################################################

    @benchmark(5)
    def count_rBrackets(base_url):
        return len(re.findall('[()]', base_url))


    ########################################################################################################################
    #               Count comma ( ',' ) symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_comma(base_url):
        return base_url.count(',')


    ########################################################################################################################
    #               Count dollar ($) symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_dollar(base_url):
        return base_url.count('$')


    ########################################################################################################################
    #               Having semicolumn (;) symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_semicolumn(url):
        return url.count(';')


    ########################################################################################################################
    #               Count (space, %20) symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_space(base_url):
        return base_url.count(' ') + base_url.count('%20')


    ########################################################################################################################
    #               Count and (&) symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_and(base_url):
        return base_url.count('&')


    ########################################################################################################################
    #               Count redirection (//) symbol at full url
    ########################################################################################################################

    @benchmark(5)
    def count_double_slash(full_url):
        list = [x.start(0) for x in re.finditer('//', full_url)]
        if list[len(list) - 1] > 6:
            return 1
        else:
            return 0


    ########################################################################################################################
    #               Count slash (/) symbol at full url
    ########################################################################################################################

    @benchmark(5)
    def count_slash(full_url):
        return full_url.count('/')


    ########################################################################################################################
    #               Count equal (=) symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_equal(base_url):
        return base_url.count('=')


    ########################################################################################################################
    #               Count percentage (%) symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_percentage(base_url):
        return base_url.count('%')


    ########################################################################################################################
    #               Count question mark (?) symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_question(base_url):
        return base_url.count('?')


    ########################################################################################################################
    #               Count underscore (_) symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_underscore(base_url):
        return base_url.count('_')


    ########################################################################################################################
    #               Count dash (-) symbol at base url
    ########################################################################################################################

    @benchmark(5)
    def count_hyphens(base_url):
        return base_url.count('-')


    ########################################################################################################################
    #              Count number of dots in hostname
    ########################################################################################################################

    @benchmark(5)
    def count_dots(hostname):
        return hostname.count('.')


    ########################################################################################################################
    #              Count number of colon (:) symbol
    ########################################################################################################################

    @benchmark(5)
    def count_colon(url):
        return url.count(':')


    ########################################################################################################################
    #               Count number of stars (*) symbol
    ########################################################################################################################

    @benchmark(5)
    def count_star(url):
        return url.count('*')


    ########################################################################################################################
    #               Count number of OR (|) symbol
    ########################################################################################################################

    @benchmark(5)
    def count_or(url):
        return url.count('|')


    ########################################################################################################################
    #               Having multiple http or https in url path
    ########################################################################################################################

    @benchmark(5)
    def count_http_token(url_path):
        return url_path.count('http')


    ########################################################################################################################
    #               Uses https protocol
    ########################################################################################################################

    @benchmark(5)
    def https_token(scheme):
        if scheme == 'https':
            return 0
        return 1


    ########################################################################################################################
    #               Ratio of digits in hostname
    ########################################################################################################################

    @benchmark(5)
    def ratio_digits(hostname):
        return len(re.sub("[^0-9]", "", hostname)) / len(hostname)


    ########################################################################################################################
    #               Count number of digits in domain/subdomain/path
    ########################################################################################################################

    @benchmark(5)
    def count_digits(line):
        return len(re.sub("[^0-9]", "", line))


    ########################################################################################################################
    #              Checks if tilde symbol exist in webpage URL
    ########################################################################################################################

    @benchmark(5)
    def count_tilde(full_url):
        if full_url.count('~') > 0:
            return 1
        return 0


    ########################################################################################################################
    #               number of phish-hints in url path
    ########################################################################################################################

    @benchmark(5)
    def phish_hints(url_path, phish_hints):
        count = 0
        for hint in [item for sublist in phish_hints.values() for item in sublist]:
            count += url_path.lower().count(hint)
        return count


    ########################################################################################################################
    #               Check if TLD exists in the path
    ########################################################################################################################

    @benchmark(5)
    def tld_in_path(tld, path):
        if path.lower().count(tld) > 0:
            return 1
        return 0


    ########################################################################################################################
    #               Check if tld is used in the subdomain
    ########################################################################################################################

    @benchmark(5)
    def tld_in_subdomain(tld, subdomain):
        if subdomain.count(tld) > 0:
            return 1
        return 0


    ########################################################################################################################
    #               Check if TLD in bad position
    ########################################################################################################################

    @benchmark(5)
    def tld_in_bad_position(tld, subdomain, path):
        if tld_in_path(tld, path)[0] == 1 or tld_in_subdomain(tld, subdomain)[0] == 1:
            return 1
        return 0


    ########################################################################################################################
    #               Abnormal subdomain starting with wwww-, wwNN
    ########################################################################################################################

    @benchmark(5)
    def abnormal_subdomain(url):
        if re.search('(http[s]?://(w[w]?|\d))([w]?(\d|-))', url):
            return 1
        return 0


    ########################################################################################################################
    #               Number of redirection
    ########################################################################################################################

    @benchmark(5)
    def count_redirection(page):
        return len(page.history)


    ########################################################################################################################
    #               Number of redirection to different domains
    ########################################################################################################################

    @benchmark(5)
    def count_external_redirection(page, domain):
        count = 0
        if len(page.history) == 0:
            return 0
        else:
            for i, response in enumerate(page.history, 1):
                if domain.lower() not in response.url.lower():
                    count += 1
            return count


    ########################################################################################################################
    #               Is the registered domain created with random characters
    ########################################################################################################################

    @benchmark(5)
    def random_domain(second_level_domain):
        return int(len([word for word in segment(second_level_domain) if word not in WORDS + brand_list]) > 0)


    ###############################tld_in_path#########################################################################################
    #               Presence of words with random characters
    ########################################################################################################################

    @benchmark(15)
    def random_words(words_raw):
        return len([word for str in [segment(word) for word in words_raw] for word in str if
                word not in WORDS + brand_list])


    ########################################################################################################################
    #               Consecutive Character Repeat
    ########################################################################################################################

    @benchmark(5)
    def char_repeat(words_raw):
        count = 0

        for word in words_raw:
            repeated = False

            if word:
                for i in range(len(word)-1):
                    if word[i] == word[i+1]:
                        if not repeated:
                            repeated = True
                            count += 1
                    else:
                        repeated = False
        return count


    ########################################################################################################################
    #               puny code in domain
    ########################################################################################################################

    @benchmark(5)
    def punycode(url):
        if url.startswith("http://xn--") or url.startswith("http://xn--"):
            return 1
        else:
            return 0


    ########################################################################################################################
    #               domain in brand list
    ########################################################################################################################


    import Levenshtein

    @benchmark(20)
    def domain_in_brand(second_level_domain):
        word = second_level_domain.lower()

        for idx, b in enumerate(brand_list):
            dst = len(Levenshtein.editops(word, b.lower()))
            if dst == 0:
                return 1 - idx / len(brand_list)
            elif dst <= (len(word)-2)/3+1:
                return 1 - idx / (len(brand_list) * 2)
        return 0


    ########################################################################################################################
    #               brand name in path
    ########################################################################################################################


    @benchmark(20)
    def brand_in_path(second_level_domain, path):
        for idx, b in enumerate(brand_list):
            if b in path and b not in second_level_domain:
                return 1 - idx / len(brand_list)
        return 0


    ########################################################################################################################
    #               count www in url words
    ########################################################################################################################

    @benchmark(5)
    def check_www(words_raw):
        count = 0
        for word in words_raw:
            if not word.find('www') == -1:
                count += 1
        return count


    ########################################################################################################################
    #               count com in url words
    ########################################################################################################################

    @benchmark(5)
    def check_com(words_raw):
        count = 0
        for word in words_raw:
            if not word.find('com') == -1:
                count += 1
        return count


    ########################################################################################################################
    #               check port presence in domain
    ########################################################################################################################

    @benchmark(5)
    def port(url):
        if re.search(
                "^[a-z][a-z0-9+\-.]*://([a-z0-9\-._~%!$&'()*+,;=]+@)?([a-z0-9\-._~%]+|\[[a-z0-9\-._~%!$&'()*+,;=:]+\]):([0-9]+)",
                url):
            return 1
        return 0


    ########################################################################################################################
    #               length of raw word list
    ########################################################################################################################

    @benchmark(5)
    def length_word_raw(words_raw):
        return len(words_raw)


    ########################################################################################################################
    #               count average word length in raw word list
    ########################################################################################################################

    @benchmark(5)
    def average_word_length(words_raw):
        if len(words_raw) == 0:
            return 0
        return sum(len(word) for word in words_raw) / len(words_raw)


    ########################################################################################################################
    #               longest word length in raw word list
    ########################################################################################################################

    @benchmark(5)
    def longest_word_length(words_raw):
        if len(words_raw) == 0:
            return 0
        return max(len(word) for word in words_raw)


    ########################################################################################################################
    #               shortest word length in raw word list
    ########################################################################################################################

    @benchmark(5)
    def shortest_word_length(words_raw):
        if len(words_raw) == 0:
            return 0
        return min(len(word) for word in words_raw)


    ########################################################################################################################
    #               prefix suffix
    ########################################################################################################################

    @benchmark(5)
    def prefix_suffix(url):
        if re.findall(r"https?://[^\-]+-[^\-]+/", url):
            return 1
        else:
            return 0

    ########################################################################################################################
    #               count subdomain
    ########################################################################################################################

    @benchmark(5)
    def count_subdomain(url):
        return len(re.findall("\.", url))


    ########################################################################################################################
    #               header server
    ########################################################################################################################


    def header_server(request):
        return request.headers['server']


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


    @benchmark(20)
    def count_visual_similarity_domains(second_level_domain):
        finded = {}

        for s, lst in symbols.items():
            shift = 0
            while True:
                idx = second_level_domain.find(s, shift)

                if idx < 0:
                    break

                shift = idx + 1
                finded[idx] = {'from': s, 'to': lst}

        dic = []
        pattern = second_level_domain.lower()

        for key in sorted(finded.keys()):
            pattern = pattern.replace(finded[key]['from'], '{}')
            dic.append(finded[key]['to'])

        N = len(dic)
        dividers = [reduce(mul, [len(dic[N - y - 1]) for y in range(N - x - 1)], 1) for x in range(N)]
        max_size = reduce(mul, [len(dic[x]) for x in range(N)], 1) - 1
        exp = '|'.join([pattern.format(*[dic[x][int((i + 1) / dividers[x]) % len(dic[x])] for x in range(N)]) for i in range(max_size)])

        if exp:
            doc = ' '.join(brand_list)
            return len(re.findall(exp, doc))
        else:
            return 0
