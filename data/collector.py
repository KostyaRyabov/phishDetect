import pandas
import requests
import io
from datetime import date
import os
from tools import benchmark
from googletrans import Translator

translator = Translator()

legitimate_url_list = []
phish_url_list = []
brand_list = []


def loadRawData():
    global brand_list, legitimate_url_list, phish_url_list

    legitimate_url_list = pandas.read_csv("data/urls/legitimate/18-01-2021.csv", header=None)[0].tolist()
    brand_list = [brand.split('.')[0] for brand in
                  pandas.read_csv("data/ranked_domains/14-1-2021.csv", header=None)[1].tolist()]

    try:
        http_request = requests.get('http://data.phishtank.com/data/online-valid.csv')
        phish_url_list = pandas.read_csv(io.StringIO(http_request.content.decode('utf-8')), usecols=['url'])
        filename = "data/urls/phish/{0}.csv".format(date.today().strftime("%d-%m-%Y"))
        phish_url_list.to_csv(filename, index=False, header=False)
        phish_url_list = phish_url_list['url'].tolist()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)


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