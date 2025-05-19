import re
import logging
import polars as pl
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.middleware import Nltk
import os 

# TODO: cleanify
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    full_handler = logging.FileHandler(f"{log_dir}/PRE.hve.log", mode='w')
    full_handler.setFormatter(formatter)
    logger.addHandler(full_handler)
    return logger

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.metrics import edit_distance


class SVCPreprocessor:
    def __init__(self, frame, column="component"):
        self.column = column
        self.cleaned_values = self._clean(frame[column])

    def _clean(self, series):
        cleaned = series.cast(str).str.to_lowercase()
        cleaned = cleaned.str.replace_all(r"[()\[\],]", " ").str.strip_chars()
        return cleaned.unique().drop_nulls().to_list()

class HVCProcessor:
    def __init__(self, values):
        self.original_values = values
        self.cleaned_units = self._split_and_clean(values)

    def _split_and_clean(self, values):
        units = []
        for val in values:
            if not isinstance(val, str) or not val.strip():
                continue
            clean_val = re.sub(r"[()\[\],]", " ", val.lower())
            clean_val = re.sub(r"\s*\|\s*", " or ", clean_val)
            clean_val = re.sub(r"\s+", " ", clean_val).strip()
            subunits = re.split(r'\b(?:and|or)\b', clean_val)
            subunits = [s.strip() for s in subunits if s.strip()]
            units.extend(subunits)
        return units

class Runner:
    def __init__(self, sigs_path, indications_path, svc_col="component", nltk_path=None, log_dir="."):
        self.sigs_path = sigs_path
        self.indications_path = indications_path
        self.svc_col = svc_col
        self.nltk_path = nltk_path
        self.s = None
        self.i = None
        self.svc_processed = None
        self._load_data()
        if nltk_path:
            self._init_nltk()
        self.logger = setup_logging(log_dir)

    def _init_nltk(self):
        print("[INFO] Initializing NLTK library...")
        self.nltk = Nltk(self.nltk_path)
        self.nltk.configure()

    def _load_data(self):
        self.s = pl.read_csv(self.sigs_path, infer_schema_length=2000)
        self.i = pl.read_csv(self.indications_path, infer_schema_length=2000, encoding="ISO-8859-1")
        self.svc_processed = SVCPreprocessor(self.s, column=self.svc_col)
        print("[INFO] Data loaded.")

    # TODO: might look greedy - implement nltk fuzzy matcher, but will work for now...
    def val2col(self, column, frame, value, thr=0.02):
        """scoring type - binary"""
        df_pl = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
        col_values = df_pl[column].drop_nulls().unique().to_list()
        all_texts = [str(value)] + [str(x) for x in col_values]
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3)) 
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
        pp = [similarities[i] for i in range(len(col_values)) if similarities[i] > thr]
        return {value: round(np.mean(pp), 2) if pp else 0.0}, pp

    # TODO: Define/USE in ETL
    def nltk_fuzzy_matcher(hvc_string, known_components, max_distance=2, lowercase=True, debug=False):
        """
        Token-level fuzzy matcher using NLTK.

        This function attempts to extract known drug/component names from a given HVC (e.g., condition, note, etc.)
        using simple tokenization, lemmatization, and edit-distance matching.

        Steps:
        1. Tokenize the input string using NLTK's word_tokenize.
        2. Lemmatize each token using WordNetLemmatizer.
        3. Compare each token against known component names (also normalized) using edit_distance.
        4. Return a set of component names where edit_distance <= max_distance.

        Args:
            hvc_string (str): A sentence or phrase from an HVC column.
            known_components (list[str]): A list of normalized component/drug names to match against.
            max_distance (int): Maximum edit distance allowed for a match (default = 2).
            lowercase (bool): Whether to normalize casing (recommended).
            debug (bool): If True, print debug matching info.

        Returns:
            matched_terms (set[str]): A set of matched component names found via fuzzy edit-distance.

        Example:
            nltk_fuzzy_matcher("2+ including lenalidomide", ["cisplatin", "lenalidomide", "paclitaxel"])
            → {"lenalidomide"}
        """
        if not isinstance(hvc_string, str) or not hvc_string.strip():
            return set()

        if lowercase:
            hvc_string = hvc_string.lower()
            known_components = [c.lower() for c in known_components]

        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(hvc_string)

        matched = set()
        for token in tokens:
            lemma = lemmatizer.lemmatize(token)
            for comp in known_components:
                dist = edit_distance(lemma, comp)
                if dist <= max_distance:
                    matched.add(comp)
                    if debug:
                        print(f"[MATCH] Token: '{token}' → Lemma: '{lemma}' ≈ '{comp}' (distance={dist})")
                elif debug:
                    print(f"[SKIP] Token: '{token}' → Lemma: '{lemma}' ≠ '{comp}' (distance={dist})")
        return matched


    def extract_matches(self, threshold=0.8, skip_words=None, skip_hvc=None):
        print("[INFO] Starting HVC extraction and matching...")
        matched_all = {}

        corrections = {
            "palictaxel": "paclitaxel",
            "alfa": "alpha"  # add more as needed
        }

        for hvc_col in self.i.columns:
            if skip_hvc and hvc_col.lower() in skip_hvc:
                msg = f"[INFO] Skipping HVC column: {hvc_col}"
                print(msg)
                continue

            print(f"[INFO] Processing HVC column: {hvc_col}")
            col_values = self.i[hvc_col].drop_nulls().to_list()
            value_to_indices = {}

            for idx, val in enumerate(col_values):
                if not isinstance(val, str):
                    continue
                cleaned = val.lower().strip()
                value_to_indices.setdefault(cleaned, []).append(idx)

            processor = HVCProcessor(list(value_to_indices.keys()))
            logs = []
            matched = {}

            for val in processor.cleaned_units:
                if val in skip_words or val.isdigit() or re.fullmatch(r"\d+\+?", val):
                    logs.append(f"{hvc_col} | {val} → skipped 🚫 (in skip list or numeric)\n")
                    continue

                val_corrected = corrections.get(val, val)
                score, _ = self.val2col(self.svc_col, self.s, val_corrected, thr=threshold)
                score_val = score.get(val_corrected, 0)

                matched_rows = []

                if score_val >= threshold:
                    for orig_val, indices in value_to_indices.items():
                        if val in orig_val:
                            matched_rows.extend([(i, val_corrected) for i in indices])
                    freq = len(set(idx for idx, _ in matched_rows))
                    logs.append(f"{hvc_col} | {val} → match ✅ (score: {score_val:.3f}, repeats: {freq})\n")
                else:
                    # Try each word separately
                    for word in val.split():
                        word = corrections.get(word, word)
                        if word in skip_words:
                            continue
                        score_single, _ = self.val2col(self.svc_col, self.s, word, thr=threshold)
                        score_word = score_single.get(word, 0)
                        if score_word >= threshold:
                            for orig_val, indices in value_to_indices.items():
                                if val in orig_val:
                                    matched_rows.extend([(i, word) for i in indices])
                            logs.append(f"{hvc_col} | {val} → partial match ✅ on '{word}' (score: {score_word:.3f})\n")
                            break
                    else:
                        logs.append(f"{hvc_col} | {val} → no match ❌ (score: {score_val:.3f})\n")

                for tup in matched_rows:
                    matched.setdefault(hvc_col, []).append(tup)

            matched_all[hvc_col] = {}
            for tup in matched.get(hvc_col, []):
                matched_all[hvc_col].setdefault(tup[1], []).append(tup[0])

            done_msg = f"[INFO] Done with {hvc_col}: {sum(len(v) for v in matched_all[hvc_col].values())} matches recorded."
            print(done_msg)
            self.logger.info(done_msg)
            self.logger.info("".join(logs))

        print("[INFO] HVC extraction complete.")
        return matched_all

    def explode_matches(self, base_df, matched_terms_dict):
        df = base_df.clone()
        exploded = []

        for hvc_col, term_to_indices in matched_terms_dict.items():
            for term, indices in term_to_indices.items():
                for idx in indices:
                    try:
                        row = df[idx].to_dict()
                    except Exception:
                        continue
                    new_row = row.copy()
                    new_row[self.svc_col] = str(term)
                    exploded.append({k: str(v) if v is not None else "" for k, v in new_row.items()})

        df_initial_shape = df.shape
        exploded_df = pl.DataFrame(exploded) if exploded else pl.DataFrame(schema=df.schema)
        updated = df.vstack(exploded_df)

        added = updated.shape[0] - df_initial_shape[0]
        expected = len(exploded)
        assert added == expected, f"Mismatch: expected {expected}, added {added}"

        stats = f"[STATS] Exploded rows: {expected} | Before: {df_initial_shape} | After: {updated.shape}"
        print(stats)
        self.logger.info(stats)

        return updated


def pre_run(
    input_files_dir=".",
    output_dir=".",
    nltk_base_path="path/to/nltk/data",
    log_dir= "log_dir"
):
    sigs_path = f"{input_files_dir}/sigs.csv"
    indications_path = f"{input_files_dir}/indications.csv"

    print("[INFO] Starting preprocessing run...")
    r = Runner(
        sigs_path=sigs_path,
        indications_path=indications_path,
        svc_col="component",
        nltk_path=nltk_base_path,
        log_dir=log_dir
    )

    matches = r.extract_matches(
        threshold=0.8,
        skip_hvc={"string", "study", "component","regulator", "date", "prior_therapy"},
        skip_words={
            "aromatase inhibitor",
            "surgery", "iodine", "ros1", "and", "or", "with", "egfr tki", "one", "other", "inhibitor", "alk",
            "resection", "chemotherapy", "radiotherapy", "platinum", "regimen", "therapy", "not", "applicable",
            "surgical castration", "androgen receptor inhibitor", "steroids", "antiestrogen",
            "anticoagulation", "contraindication to cisplatin", "cisplatin-ineligible",
            "1+ including auto hsct unless ineligible", "autologous hsct", "immunomodulator", "anti-cd38 antibody"
        }
    )

    updated_df = r.explode_matches(r.i, matches)
    r.s.to_pandas().to_csv(f"{output_dir}/s_frame.tsv", sep="\t", index=False)
    updated_df.to_pandas().to_csv(f"{output_dir}/i_frame.tsv", sep="\t", index=False)
    print("[INFO] Output files written.")

if __name__ == "__main__":
    pre_run()