from main import state

if state == 0:
    def download_phishURLS():
        try:
            http_request = requests.get('http://data.phishtank.com/data/online-valid.csv')
            phish_url_list = pandas.read_csv(io.StringIO(http_request.content.decode('utf-8')), usecols=['url'])
            filename = "data/urls/phish/{0}.csv".format(date.today().strftime("%d-%m-%Y"))
            phish_url_list.to_csv(filename, index=False, header=False)
            phish_url_list = phish_url_list['url'].tolist()
        except requests.exceptions.RequestException as e:
            raise SystemExit('Access denied [PHISHTANK]')

        return phish_url_list


if state in range(1, 7):
    import pandas
    import requests
    import io
    from datetime import date
    import os


    def get_dir_path():
        lst = os.listdir(os.getcwd() + '/data/datasets/RAW')
        if lst:
            n = int(max(list(map(int, lst)))) + 1
            dir_path = "data/datasets/RAW/{}/".format(n)
            os.mkdir(os.getcwd() + '/' + dir_path)
        else:
            dir_path = "data/datasets/RAW/1/"
            os.mkdir(os.getcwd() + '/' + dir_path)

        return dir_path


    dir_path = get_dir_path()


    brand_list = [brand.split('.')[0] for brand in
                      pandas.read_csv("data/ranked_domains/14-1-2021.csv", header=None)[1].tolist()][:100000]


    def set_lable_to_list(lst, lable):
        return [(url, lable) for url in lst]

    def load_legitimateURLS(date):
        legitimate_url_list = pandas.read_csv("data/urls/legitimate/{}.csv".format(date), header=None)[0].tolist()
        return legitimate_url_list

    def load_phishURLS(date):
        phish_url_list = pandas.read_csv("data/urls/phish/{}.csv".format(date), header=None)[0].tolist()
        return phish_url_list