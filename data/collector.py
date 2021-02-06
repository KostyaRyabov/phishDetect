import pandas
import requests
import io
from datetime import date


brand_list = [brand.split('.')[0] for brand in
                  pandas.read_csv("data/ranked_domains/14-1-2021.csv", header=None)[1].tolist()]


def set_lable_to_list(lst, lable):
    return [(url, lable) for url in lst]


def loadRawData():
    legitimate_url_list = []
    phish_url_list = []

    legitimate_url_list = pandas.read_csv("data/urls/legitimate/18-01-2021.csv", header=None)[0].tolist()

    # try:
    #     http_request = requests.get('http://data.phishtank.com/data/online-valid.csv')
    #     phish_url_list = pandas.read_csv(io.StringIO(http_request.content.decode('utf-8')), usecols=['url'])
    #     filename = "data/urls/phish/{0}.csv".format(date.today().strftime("%d-%m-%Y"))
    #     phish_url_list.to_csv(filename, index=False, header=False)
    #     phish_url_list = phish_url_list['url'].tolist()
    # except requests.exceptions.RequestException as e:
    #     raise SystemExit(e)

    return legitimate_url_list, phish_url_list