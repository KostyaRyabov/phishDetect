import pandas
import requests
import io
from datetime import date
import os


def get_dir_path():
    lst = os.listdir(os.getcwd() + '/data/datasets')
    if lst:
        n = int(max(list(map(int, lst)))) + 1
        dir_path = "data/datasets/{}/".format(n)
        os.mkdir(os.getcwd() + '/' + dir_path)
    else:
        dir_path = "data/datasets/1/"
        os.mkdir(os.getcwd()+'/'+dir_path)

    return dir_path


dir_path = get_dir_path()


brand_list = [brand.split('.')[0] for brand in
                  pandas.read_csv("data/ranked_domains/14-1-2021.csv", header=None)[1].tolist()]


def set_lable_to_list(lst, lable):
    return [(url, lable) for url in lst]


def download_phishURLS():
    try:
        http_request = requests.get('http://data.phishtank.com/data/online-valid.csv')
        phish_url_list = pandas.read_csv(io.StringIO(http_request.content.decode('utf-8')), usecols=['url'])
        filename = "data/urls/phish/{0}.csv".format(date.today().strftime("%d-%m-%Y"))
        phish_url_list.to_csv(filename, index=False, header=False)
        phish_url_list = phish_url_list['url'].tolist()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

    return phish_url_list


def load_legitimateURLS():
    legitimate_url_list = pandas.read_csv("data/urls/legitimate/30-01-2021.csv", header=None)[0].tolist()
    return legitimate_url_list

def load_phishURLS():
    phish_url_list = pandas.read_csv("data/urls/phish/08-02-2021.csv", header=None)[0].tolist()
    return phish_url_list