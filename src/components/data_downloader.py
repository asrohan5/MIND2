import os
import requests
import zipfile
import io
import sys
from src.logger import logging
from src.exception import CustomException

class MINDDataDownloader:
    def __init__(self, download_url, extract_path="D:/Projects/MIND2/artifacts/raw"):
        self.download_url = download_url
        self.extract_path = extract_path

    def download_and_extract(self):
        try:
            os.makedirs(self.extract_path, exist_ok=True)
            logging.info(f"Downloading dataset from {self.download_url}")

            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()

            logging.info("Download complete. Extracting zip.")

            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(self.extract_path)

            logging.info(f"Extraction complete. Files are in: {self.extract_path}")

        except Exception as e:
            raise CustomException(e, sys)
