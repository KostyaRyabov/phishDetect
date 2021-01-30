import feature_extractor as fe
from data import collector


if __name__ == "__main__":
    fe.generate_legitimate_urls(20000)
    # collector.loadRawData()

    # fe.URLs_analyser(legitimate_url_list)