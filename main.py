import feature_extractor as fe
from data.collector import loadRawData, set_lable_to_list


if __name__ == "__main__":
    # fe.generate_legitimate_urls(20000)

    legitimate_url_list, phish_url_list = loadRawData()

    url_list = set_lable_to_list(legitimate_url_list, 0) + set_lable_to_list(phish_url_list, 1)

    fe.generate_dataset(url_list[:4])
    # fe.extract_features(url_list[2][0], url_list[2][1])