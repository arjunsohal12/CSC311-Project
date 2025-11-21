from __future__ import annotations
import pandas as pd
from typing import List, Optional, Union
import numpy as np
import string

TABLE = str.maketrans('', '', string.punctuation)

numeric_cols = [
    'How likely are you to use this model for academic tasks?',
    'How often do you expect this model to provide responses with references or supporting evidence?',
    "How often do you verify this model's responses?",
    'Based on your experience, how often has this model given you a response that felt suboptimal?',

]
text_cols = [
    "In your own words, what kinds of tasks would you use this model for?",
    'Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?',
    "When you verify a response from this model, how do you usually go about it?",
]
hot_cols = [
    'Which types of tasks do you feel this model handles best? (Select all that apply.)',
    'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)',
]
class DataReader:
    """
    Lightweight CSV reader/inspector built on pandas.
    """

    def __init__(self, path: str, back_compat = True, **read_csv_kwargs):
        """
        path: path to the CSV file
        read_csv_kwargs: any pandas.read_csv keyword args (e.g., sep=',', encoding='utf-8')
        """
        self.path = path
        self.read_csv_kwargs = read_csv_kwargs
        self.load(back_compat=back_compat)
    # ---------- I/O ----------
    def to_numpy(self):
        blocks = []

        for col in hot_cols:
            blocks.append(np.stack(self.X[col].to_numpy(), axis = 0)) # (N, d) for each
        for col in numeric_cols:
            blocks.append(np.expand_dims(self.X[col].to_numpy(), axis = 1)) # (N, d)
        for col in text_cols:
            blocks.append(np.stack(self.X[col].to_numpy(), axis = 0)) # (N, d) for each
        # blocks.append(self.embeddings)  # (N, 384)
        # blocks.append(self.tfidf_matrix)  # shape (N, V)

        X = np.concatenate(blocks, axis = -1) # (N, total_d)
        Y = self.labels.to_numpy()

        return X, Y
    def load(self, back_compat = True) -> "DataReader":
        """Load (or reload) and preprocess the CSV into memory."""
        if not back_compat:
            self.X = pd.read_csv(self.path, **self.read_csv_kwargs)

            self.X = self.X[self.X["label"].notna()]

            for col in text_cols:
                if col in self.X.columns:
                    self.X[col] = self.X[col].fillna("")

            for col in numeric_cols:
                if col in self.X.columns:
                    self.X[col] = self.X[col].fillna("-1")

            self.labels = self.X.pop("label")

            if "student_id" in self.X.columns:
                self.X = self.X.drop("student_id", axis=1)
        else:
            self.X = pd.read_csv(self.path, **self.read_csv_kwargs)
            self.X.dropna(inplace=True)
            self.labels = self.X.pop("label")
            self.X = self.X.drop('student_id', axis = 1)

        combined_text = (
            self.X[text_cols]
            .fillna("")
            .agg(" ".join, axis=1)
        )
        # from sentence_transformers import SentenceTransformer

        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # embeddings = model.encode(combined_text.tolist(), show_progress_bar=True)
        # self.embeddings = embeddings  # shape (N, 384)

        # tfidf = TfidfVectorizer(
        #     max_features=3000,        # cap vocab size to prevent explosion
        #     ngram_range=(1, 1),       # unigrams + bigrams (often improves performance)
        #     min_df=5                  # ignore extremely rare words
        # )
        # self.tfidf_matrix = tfidf.fit_transform(combined_text).toarray()  # (N, V)

        for col in numeric_cols:
            self.X[col] = self.X[col].str.split(" ").str[0].astype(int)
        
        for col in hot_cols:
            selections = set()
            for cell in self.X[col].dropna():
                for item in cell.split(","):
                    selections.add(item.strip())
            selections = sorted(selections)
            selections = [x for x in selections if len(x) >= len('Math computations')]
            # print(selections)

            def to_vector(cell):
                tokens = {x.strip() for x in str(cell).split(",") if x.strip()}
                return [1 if cat in tokens else 0 for cat in selections]

            self.X[col] = self.X[col].fillna("").apply(to_vector)

        for col in text_cols:

            def remove_punctuation(cell):
                clean = cell.translate(TABLE).lower()
                return clean
            self.X[col] = self.X[col].fillna("").apply(remove_punctuation)

            selections = set()
            for cell in self.X[col].dropna():
                for item in cell.split(" "):
                    selections.add(item.strip())
            selections = sorted(selections)

            def to_vector(cell):
                tokens = {x.strip() for x in str(cell).split(" ") if x.strip()}
                return [1 if cat in tokens else 0 for cat in selections]
            
            self.X[col] = self.X[col].fillna("").apply(to_vector)

        return self

    # ---------- Quick views ----------
    def header(self) -> List[str]:
        """Return column names."""
        self._ensure_loaded()
        return list(self.X.columns)

    def show(self, n: int = 5) -> pd.DataFrame:
        """Return the first n rows."""
        self._ensure_loaded()
        return self.X.head(n)

    def show_tail(self, n: int = 5) -> pd.DataFrame:
        """Return the last n rows."""
        self._ensure_loaded()
        return self.X.tail(n)

    def sample(self, n: int = 5, random_state: Optional[int] = None) -> pd.DataFrame:
        """Return a random sample of n rows."""
        self._ensure_loaded()
        return self.X.sample(n=n, random_state=random_state)

    # ---------- Structure & summary ----------
    def shape(self) -> tuple[int, int]:
        """(rows, columns)."""
        self._ensure_loaded()
        return self.X.shape

    def info(self) -> str:
        """Return DataFrame info as a string."""
        self._ensure_loaded()
        buf = []
        self.X.info(buf=buf)
        return "\n".join(buf)

    def describe(self, include: Union[str, List[str]] = "all") -> pd.DataFrame:
        """Descriptive stats; include='all' to summarize mixed dtypes."""
        self._ensure_loaded()
        return self.X.describe(include=include)

    def missing_summary(self) -> pd.Series:
        """Count missing values per column."""
        self._ensure_loaded()
        return self.X.isna().sum().sort_values(ascending=False)

    # ---------- Column/row helpers ----------
    def columns(self) -> List[str]:
        """Alias for header()."""
        return self.header()

    def get(self, col: str) -> pd.Series:
        """Return a column by name."""
        self._ensure_loaded()
        return self.X[col]

    def value_counts(self, col: str, dropna: bool = True) -> pd.Series:
        """Value counts for a column."""
        self._ensure_loaded()
        return self.X[col].value_counts(dropna=dropna)

    def select(self, cols: List[str]) -> pd.DataFrame:
        """Select a subset of columns."""
        self._ensure_loaded()
        return self.X[cols]

    # ---------- Internal ----------
    def _ensure_loaded(self):
        if self.X is None:
            raise RuntimeError("Data not loaded. Call .load() first.")

# ---------------- Example ----------------
if __name__ == "__main__":
    dr = DataReader("training_data_clean.csv")
    # print(dr.X.columns)
    # print(dr.X.loc[0])
    x, y = dr.to_numpy()
    print(x.shape)
    print(y.shape)
    # print(dr.X.loc[0]['Which types of tasks do you feel this model handles best? (Select all that apply.)'])
