import pandas as pd
from pathlib import Path
import nltk
from functools import wraps
import logging
import shutil
import tempfile
import threading
from nltk.data import find

class Nltk:
    def __init__(self, base_path="."):
        self.nltk_data_path = Path(base_path) / "nltk_tree" / "nltk_data"
        self.required_resources = ["words", "punkt", "wordnet", "omw-1.4"]

    def configure(self):
        """Sets the NLTK data path and ensures all required corpora/tokenizers are available."""
        self.nltk_data_path.mkdir(parents=True, exist_ok=True)
        
        nltk.data.path.append(str(self.nltk_data_path))

        for resource in self.required_resources:
            try:
                find(resource)
            except LookupError:
                print(f"[INFO] NLTK resource '{resource}' not found.")
                print(f"[INFO] Downloading '{resource}'...")
                nltk.download(resource, download_dir=str(self.nltk_data_path))

        print(f"[INFO] NLTK configured with data path: {self.nltk_data_path}")

    def get_word_set(self):
        """Returns the word set from the configured NLTK corpus."""
        return set(nltk.corpus.words.words())


class DataProcessor:
    def __init__(self, logdir=".logs"):
        """
        Initializes logging and handles CSV encoding detection.

        Args:
            logdir (str): Directory for log files.
        """
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.logger = self._set_log_conf()

    def _set_log_conf(self):
        """Set up logging configuration."""
        log_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s")
        log_handler = logging.FileHandler(self.logdir / "encoding_errors.log")
        log_handler.setFormatter(log_formatter)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(log_handler)
        return logger

    def load_encoded(self, dataframe_path: str, sep=","):
        """
        Loads a CSV file with automatic encoding detection.

        Args:
            dataframe_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame or None if loading fails.
        """
        dataframe_path = Path(dataframe_path)

        if not dataframe_path.exists():
            self.logger.error(f"❌ File not found: {dataframe_path}")
            return None

        print(f"[INFO] Decoding: {dataframe_path.name}")

        success = False
        df = None

        for encoding in ["utf-8", "ISO-8859-1", "Windows-1252"]:
            try:
                df = pd.read_csv(dataframe_path, encoding=encoding, sep=sep)
                self.logger.info(f"Successfully processed: {dataframe_path.name} with encoding {encoding}")
                success = True
                break
            except UnicodeDecodeError:
                self.logger.warning(f"Encoding error: {dataframe_path.name} failed with {encoding}")
            except Exception as e:
                self.logger.error(f"Unexpected error on {dataframe_path.name}: {e}")
                break

        if not success:
            self.logger.error(f"Skipping {dataframe_path.name} due to encoding issues")

        return df

    
def ensure_encoded(func):
    """Decorator to ensure encoding detection before processing CSVs."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        def detect_and_prepare_files():
            """
            Detects `.csv` and `.tsv` files in the directory.
            If both exist, converts `.tsv` to temporary `.csv` files for consistency.
            The temporary `.csv` files will be deleted after 5 seconds.
            """
            csv_files = [f for f in self.csv_dir.rglob("*.csv")]
            sep = ","

            if not csv_files:
                csv_files = [f for f in self.csv_dir.rglob("*.tsv")]
                sep = "\t"

            try:
                assert bool(csv_files)

                tsv_files = [f for f in self.csv_dir.rglob("*.tsv")]

                if csv_files and tsv_files:
                    print("[WARN] Both CSV and TSV files exist! Converting TSV to temporary CSVs...")

                    temp_dir = Path(tempfile.mkdtemp())  # Create a temp directory
                    converted_files = []

                    for tsv_file in tsv_files:
                        temp_csv_path = temp_dir / f"{tsv_file.stem}.csv"

                        # Convert `.tsv` to `.csv`
                        with open(tsv_file, "r", encoding="utf-8") as fin, open(temp_csv_path, "w", encoding="utf-8") as fout:
                            fout.write(fin.read().replace("\t", ","))  # Replace tabs with commas
                        
                        converted_files.append(temp_csv_path)

                    csv_files = converted_files + csv_files # Use the converted temp files
                    sep = ","
                    print(f"[INFO] Converted {len(tsv_files)} TSV files to temporary CSVs.")

                    # **Schedule deletion after 5 seconds**
                    def cleanup():
                        print("[INFO] Deleting temporary files...")
                        shutil.rmtree(temp_dir, ignore_errors=True)

                    threading.Timer(5, cleanup).start()  # Cleanup after 5 seconds

            except AssertionError:
                print("[ERR] No CSV or TSV files found in the directory!")
            
            return csv_files, sep



        processor = DataProcessor() 
        self.tables = {}  # Reset tables before loading new ones
        print("finding csv recursive")

        csv_files, sep = detect_and_prepare_files()

        for file in csv_files:
            print("Reading:", file)
            encoded_df = processor.load_encoded(file, sep)  # Ensure proper encoding detection
            if encoded_df is not None:
                print("Passed:", file)
                self.tables[file.stem] = encoded_df  # Store the DataFrame directly
        return func(self, *args, **kwargs)

    return wrapper
