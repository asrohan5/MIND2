''' 
from src.components.data_downloader import MINDDataDownloader

if __name__ == "__main__":
    downloader = MINDDataDownloader(
        download_url="https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip"
    )
    downloader.download_and_extract()
'''


from src.components.data_parser import MINDDataParser

if __name__ == "__main__":
    parser = MINDDataParser(raw_data_path="D:/Projects/MIND2/artifacts/raw")

    news_df = parser.parse_news()
    print(news_df.head())

    behaviors_df = parser.parse_behaviors()
    print(behaviors_df.head())
