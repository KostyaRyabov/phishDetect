import requests
import pandas


legitimate_url_list = []
phish_url_list = []
brand_list = []


if __name__ == "__main__":
    legitimate_url_list = pandas.read_csv("datasets/legitimate/top-websites-1m (14-1-2021).csv", usecols=[1], header=None)

