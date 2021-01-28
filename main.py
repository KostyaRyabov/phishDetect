import requests
import pandas
import io

import feature_extractor as fe

legitimate_url_list = []
phish_url_list = []


def loadRawData():
    global brand_list, legitimate_url_list, phish_url_list

    legitimate_url_list = pandas.read_csv("data/urls/legitimate/18-01-2021.csv", header=None)[0].tolist()

    # try:
    #     http_request = requests.get('http://data.phishtank.com/data/online-valid.csv')
    #     phish_url_list = pandas.read_csv(io.StringIO(http_request.content.decode('utf-8')), usecols=['url'])
    #     filename = "data/urls/phish/{0}.csv".format(date.today().strftime("%d-%m-%Y"))
    #     phish_url_list.to_csv(filename, index=False, header=False)
    #     phish_url_list = phish_url_list['url'].tolist()
    # except requests.exceptions.RequestException as e:
    #     raise SystemExit(e)


if __name__ == "__main__":
    # fe.generate_legitimate_urls(10000)
    loadRawData()

    fe.URLs_analyser(legitimate_url_list)