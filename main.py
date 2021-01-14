import requests
import pandas
import io
from datetime import date


legitimate_url_list = []
phish_url_list = []
brand_list = []


def loadDatasets():
    global brand_list, legitimate_url_list, phish_url_list

    brand_list = pandas.read_csv("datasets/brands/brandirectory-ranking-data-global-2020.csv", usecols=['Brand'])[
        "Brand"].tolist()

    legitimate_url_list = pandas.read_csv("datasets/legitimate/top-websites-1m (14-1-2021).csv", usecols=[1],
                                          header=None)[1].tolist()

    try:
        http_request = requests.get('http://data.phishtank.com/data/online-valid.csv')
        phish_url_list = pandas.read_csv(io.StringIO(http_request.content.decode('utf-8')), usecols=['url'])
        filename = "datasets/phish/{0}.csv".format(date.today().strftime("%d-%m-%Y"))
        phish_url_list.to_csv(filename, index=False, header=False)
        phish_url_list = phish_url_list['url'].tolist()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)


if __name__ == "__main__":
    loadDatasets()

    for item in legitimate_url_list:
        url = "http://" + item
        print(url)
        try:
            r = requests.get(url)
            print("[+]")
        except requests.exceptions.RequestException as e:
            print("[-]")