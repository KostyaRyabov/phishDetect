from main import state
import pandas
import requests
import io
from datetime import date
import os

if state == 0:
    def download_phishURLS():
        try:
            http_request = requests.get('http://data.phishtank.com/data/online-valid.csv')
            phish_url_list = pandas.read_csv(io.StringIO(http_request.content.decode('utf-8')), usecols=['url'])
            filename = "data/urls/phish/{0}.csv".format(date.today().strftime("%d-%m-%Y"))
            phish_url_list.to_csv(filename, index=False, header=False)
            phish_url_list = phish_url_list['url'].tolist()
        except Exception as e:
            raise SystemExit('Access denied [PHISHTANK]')

        return phish_url_list


if state in range(1, 7) or state == -1:
    import pandas
    import requests
    import io
    from datetime import date
    import os


    dir_path = '/data/datasets/PROCESS/'


    brand_list = [brand.split('.')[0] for brand in
                      pandas.read_csv("data/ranked_domains/14-1-2021.csv", header=None)[1].tolist()]


    def set_lable_to_list(lst, lable):
        return [(url, lable) for url in lst]

    def load_legitimateURLS(date):
        legitimate_url_list = pandas.read_csv("data/urls/legitimate/{}.csv".format(date), header=None)[0].tolist()
        return legitimate_url_list

    def load_phishURLS(date):
        phish_url_list = pandas.read_csv("data/urls/phish/{}.csv".format(date), header=None)[0].tolist()
        return phish_url_list