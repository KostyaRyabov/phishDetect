import feature_extractor as fe
import data.collector as dc


if __name__ == "__main__":
    # fe.generate_legitimate_urls(20000)

    # dc.download_phishURLS()   # use VPN!!!

    legitimate_url_list = dc.load_legitimateURLS()
    phish_url_list = dc.load_phishURLS()
    url_list = dc.set_lable_to_list(phish_url_list, 1) + dc.set_lable_to_list(
        legitimate_url_list, 0)
    # url_list = set_lable_to_list(legitimate_url_list, 0)
    fe.generate_dataset(url_list)

    # fe.generate_dataset([('http://mail.ru', 0)])
    # fe.extract_features('http://mail.ru', 0)