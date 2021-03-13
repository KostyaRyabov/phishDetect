state = 9

# 0 - download phish_urls
# 1 - search leg urls from popular domains
# 2 - run only leg sites
# 3 - run only phish sites
# 4 - run all sites
# 5 - run example with shield and save data
# 6 - run example without shield
# 7 - collect all datasets
# 8 - select features
# 9 - generate hyperoptions for neural_networks_archSearch
# 10 - learn neural_networks
# 11 - generate hyperoptions for kNN
# 12 - generate hyperoptions for SVM
# 13 - generate hyperoptions for DT
# 14 - generate hyperoptions for ET
# 15 - generate hyperoptions for RF
# 16 - generate hyperoptions for AdaBoost_DT
# 17 - generate hyperoptions for GradientBoost
# 18 - generate hyperoptions for HistGradientBoost
# 19 - learn Gaussian_NB
# 20 - learn Bernoulli_NB
# 21 - learn Complement_NB
# 22 - learn Multinomial_NB
# 23 - get ratings
# 24 - Bagging_DT
# 25 - Stacking
# 26 - find_best_NN


import feature_extractor as fe
import data.collector as dc

if state in range(2, 6):
    seed_l = 13023
    seed_p = 1000


if __name__ == "__main__":
    if state == 0:
        dc.download_phishURLS()  # use VPN!!!
    elif state == 1:
        fe.generate_legitimate_urls(20000)
    elif state == 2:
        legitimate_url_list = dc.load_legitimateURLS('30-01-2021')
        url_list = dc.set_lable_to_list(legitimate_url_list[seed_l:], 0)
        fe.generate_dataset(url_list)
    elif state == 3:
        phish_url_list = dc.load_phishURLS('15-02-2021')
        url_list = dc.set_lable_to_list(phish_url_list[seed_p:], 1)
        fe.generate_dataset(url_list)
    elif state == 4:
        legitimate_url_list = dc.load_legitimateURLS('30-01-2021')
        phish_url_list = dc.load_phishURLS('15-02-2021')
        url_list = []
        url_list += dc.set_lable_to_list(phish_url_list[seed_p:], 1)
        url_list += dc.set_lable_to_list(legitimate_url_list[seed_l:], 0)
        fe.generate_dataset(url_list)
    elif state == 5:
        fe.generate_dataset([('http://mail.ru', 0)])
    elif state == 6:
        fe.extract_features('http://mail.ru', 0)
    elif state == 7:
        fe.combine_datasets()
    elif state == 8:
        fe.select_features(40)
    elif state == 9:
        from ml_algs import neural_networks_archSearch
        neural_networks_archSearch()
    elif state == 10:
        from ml_algs import neural_networks
        neural_networks()
    elif state == 11:
        from ml_algs import KNN
        KNN()
    elif state == 12:
        from ml_algs import SVM
        SVM()
    elif state == 13:
        from ml_algs import DT
        DT()
    elif state == 14:
        from ml_algs import ET
        ET()
    elif state == 15:
        from ml_algs import RF
        RF()
    elif state == 16:
        from ml_algs import AdaBoost_DT
        AdaBoost_DT()
    elif state == 17:
        from ml_algs import GradientBoost
        GradientBoost()
    elif state == 18:
        from ml_algs import HistGradientBoost
        HistGradientBoost()
    elif state == 19:
        from ml_algs import Gaussian_NB
        Gaussian_NB()
    elif state == 20:
        from ml_algs import Bernoulli_NB
        Bernoulli_NB()
    elif state == 21:
        from ml_algs import Complement_NB
        Complement_NB()
    elif state == 22:
        from ml_algs import Multinomial_NB
        Multinomial_NB()
    elif state == 23:
        from ml_algs import get_rating
        get_rating()
    elif state == 24:
        from ml_algs import Bagging_DT
        Bagging_DT()
    elif state == 25:
        from ml_algs import Stacking
        Stacking()
    elif state == 26:
        from ml_algs import find_best_NN
        find_best_NN(-0.01, 100)
