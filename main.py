state = 47

# 0 - download phish_urls
# 1 - search leg urls from popular domains
# 2 - run only leg sites
# 3 - run only phish sites
# 4 - run all sites
# 5 - run example with shield and save data
# 6 - run example without shield
# 7 - collect all datasets
# 8 - select features
# 9 - neural_networks_kfold
# 10 - neural_networks
# 11 - AdaBoost_DT_cv
# 12 - AdaBoost_DT
# 13 - Bagging_DT_cv
# 14 - Bagging_DT
# 15 - DT_cv
# 16 - DT
# 17 - ET_cv
# 18 - ET
# 19 - GradientBoost_cv
# 20 - GradientBoost
# 21 - HistGradientBoost_cv
# 22 - HistGradientBoost
# 23 - kNN_cv
# 24 - KNN
# 25 - logistic_regression_cv
# 26 - logistic_regression
# 27 - RF_cv
# 28 - RF
# 29 - SVM_cv
# 30 - SVM
# 31 - XGB_cv
# 32 - XGB
# 33 - Stacking (AB, RF, ET, B, HGB, GB, DT, XGB)
# 34 - Stacking (AB, RF, ET, B, XGB)
# 35 - Stacking (All)
# 36 - Stacking (ANN, LR, GB, HGB, XGB, AB)
# 37 - Stacking (GNB, BNB, CNB, MNB)
# 38 - Stacking (KNN, SVM, ANN)
# 39 - Stacking (KNN, SVM, ANN, DT, GNB, LR)
# 40 - Stacking (LR, GNB, BNB, CNB, MNB)
# 41 - Stacking (KNN, BNB, RF)

# 42 - Bernoulli_NB
# 43 - Complement_NB
# 44 - Gaussian_NB
# 45 - Multinominal_NB

# 46 - get all models

# 47 - get_rating

# 48 - search best data size


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
        fe.select_features(20)
    elif state == 9:
        from ml_algs import neural_networks_kfold
        neural_networks_kfold()
    elif state == 10:
        from ml_algs import neural_networks
        neural_networks()
    elif state == 11:
        from ml_algs import AdaBoost_DT_cv
        AdaBoost_DT_cv()
    elif state == 12:
        from ml_algs import AdaBoost_DT
        AdaBoost_DT()
    elif state == 13:
        from ml_algs import Bagging_DT_cv
        Bagging_DT_cv()
    elif state == 14:
        from ml_algs import Bagging_DT
        Bagging_DT()
    elif state == 15:
        from ml_algs import DT_cv
        DT_cv()
    elif state == 16:
        from ml_algs import DT
        DT()
    elif state == 17:
        from ml_algs import ET_cv
        ET_cv()
    elif state == 18:
        from ml_algs import ET
        ET()
    elif state == 19:
        from ml_algs import GradientBoost_cv
        GradientBoost_cv()
    elif state == 20:
        from ml_algs import GradientBoost
        GradientBoost()
    elif state == 21:
        from ml_algs import HistGradientBoost_cv
        HistGradientBoost_cv()
    elif state == 22:
        from ml_algs import HistGradientBoost
        HistGradientBoost()
    elif state == 23:
        from ml_algs import KNN_cv
        KNN_cv()
    elif state == 24:
        from ml_algs import KNN
        KNN()
    elif state == 25:
        from ml_algs import logistic_regression_cv
        logistic_regression_cv()
    elif state == 26:
        from ml_algs import logistic_regression
        logistic_regression()
    elif state == 27:
        from ml_algs import RF_cv
        RF_cv()
    elif state == 28:
        from ml_algs import RF
        RF()
    elif state == 29:
        from ml_algs import SVM_cv
        SVM_cv()
    elif state == 30:
        from ml_algs import SVM
        SVM()
    elif state == 31:
        from ml_algs import XGB_cv
        XGB_cv()
    elif state == 32:
        from ml_algs import XGB
        XGB()
    elif state == 33:
        from ml_algs import Stacking
        Stacking("AB, RF, ET, B, HGB, GB, DT, XGB")
    elif state == 34:
        from ml_algs import Stacking
        Stacking("AB, RF, ET, B, XGB")
    elif state == 35:
        from ml_algs import Stacking
        Stacking("All")
    elif state == 36:
        from ml_algs import Stacking
        Stacking("ANN, LR, GB, HGB, XGB, AB")
    elif state == 37:
        from ml_algs import Stacking
        Stacking("GNB, BNB, CNB, MNB")
    elif state == 38:
        from ml_algs import Stacking
        Stacking("KNN, SVM, ANN")
    elif state == 39:
        from ml_algs import Stacking
        Stacking("KNN, SVM, ANN, DT, GNB, LR")
    elif state == 40:
        from ml_algs import Stacking
        Stacking("LR, GNB, BNB, CNB, MNB")
    elif state == 41:
        from ml_algs import Stacking
        Stacking("KNN, BNB, RF")
    elif state == 42:
        from ml_algs import Bernoulli_NB
        Bernoulli_NB()
    elif state == 43:
        from ml_algs import Complement_NB
        Complement_NB()
    elif state == 44:
        from ml_algs import Gaussian_NB
        Gaussian_NB()
    elif state == 45:
        from ml_algs import Multinomial_NB
        Multinomial_NB()
    elif state == 46:
        from ml_algs import *

        # Bernoulli_NB()
        # Complement_NB()
        # Gaussian_NB()
        # Multinomial_NB()
        # DT()
        # AdaBoost_DT()
        # Bagging_DT()
        # ET()
        # GradientBoost()
        # HistGradientBoost()
        # KNN()
        # logistic_regression()
        # RF()
        # XGB()
        # neural_networks()
        SVM()

        Stacking("ET, B, LR")  # worst all with NB

        # Stacking("AB, GB, XGB, HGB, RF, B, ET")    # all ansambles
        # Stacking("AB, GB, XGB, HGB")  # best ansambles
        # Stacking("B, ET")  # worst ansambles

        # Stacking("GNB, CNB, MNB, BNB")  # only naive Bayesan

        # Stacking("DT, ANN, KNN")  # best models without NB
        Stacking("SVM, LR")  # worst models without NB
        # Stacking("DT, ANN, KNN, SVM, LR, GNB, CNB, MNB, BNB")  # all models with NB
        # Stacking("DT, ANN, KNN, SVM, LR")  # all models without NB

        Stacking("All")
    elif state == 47:
        from ml_algs import get_rating
        get_rating()
    elif state == 48:
        from ml_algs import search_data_size
        search_data_size()