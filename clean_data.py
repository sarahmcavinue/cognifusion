import os
import shutil
import pdfplumber
from urllib.parse import urlparse
from urllib.request import urlretrieve



class CleanData:
    def __init__(self, data_dir, urls):
        self.data_dir = data_dir
        self.urls = urls
        os.makedirs(self.data_dir, exist_ok=True)

    def sanitize_filename(self, url_or_path):
        """
        Extracts and sanitizes a filename from a URL or a local file path.
        """
        if url_or_path.startswith("http"):
            parsed_url = urlparse(url_or_path)
            filename = os.path.basename(parsed_url.path)
        else:
            filename = os.path.basename(url_or_path)

        for char in ['?', '&', '\\', '/', ':', '*']:
            filename = filename.replace(char, '_')
        return filename

    def download_files(self):
        for url_or_path in self.urls:
            sanitized_filename = self.sanitize_filename(url_or_path)
            file_path = os.path.join(self.data_dir, sanitized_filename)

            os.makedirs(self.data_dir, exist_ok=True)

            if url_or_path.startswith("http"):
                try:
                    urlretrieve(url_or_path, file_path)
                    print(f"Downloaded {url_or_path} to {file_path}")
                except Exception as e:
                    print(f"Error downloading {url_or_path}: {e}")
            else:
                if os.path.exists(url_or_path):
                    shutil.copy2(url_or_path, file_path)
                    print(f"Copied local file {url_or_path} to {file_path}")
                else:
                    print(f"Local file {url_or_path} does not exist.")

    def get_pdf_text(self):
        text = ""
        for pdf_file in os.listdir(self.data_dir):
            if pdf_file.endswith('.pdf'):
                full_path = os.path.join(self.data_dir, pdf_file)
                try:
                    with pdfplumber.open(full_path) as pdf:
                        for page in pdf.pages:
                            text += (page.extract_text() or '')
                except Exception as e:
                    print(f"Failed to process {pdf_file}: {e}")
        return text
    
    
