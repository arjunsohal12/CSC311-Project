from __future__ import annotations
import pandas as pd
from typing import List, Optional, Union

class DataReader:
    """
    Lightweight CSV reader/inspector built on pandas.
    """

    def __init__(self, path: str, **read_csv_kwargs):
        """
        path: path to the CSV file
        read_csv_kwargs: any pandas.read_csv keyword args (e.g., sep=',', encoding='utf-8')
        """
        self.path = path
        self.read_csv_kwargs = read_csv_kwargs
        self.df: pd.DataFrame = pd.read_csv(self.path, **self.read_csv_kwargs)

    # ---------- I/O ----------
    def load(self) -> "DataReader":
        """Load (or reload) the CSV into memory."""
        self.df = pd.read_csv(self.path, **self.read_csv_kwargs)
        return self

    # ---------- Quick views ----------
    def header(self) -> List[str]:
        """Return column names."""
        self._ensure_loaded()
        return list(self.df.columns)

    def show(self, n: int = 5) -> pd.DataFrame:
        """Return the first n rows."""
        self._ensure_loaded()
        return self.df.head(n)

    def show_tail(self, n: int = 5) -> pd.DataFrame:
        """Return the last n rows."""
        self._ensure_loaded()
        return self.df.tail(n)

    def sample(self, n: int = 5, random_state: Optional[int] = None) -> pd.DataFrame:
        """Return a random sample of n rows."""
        self._ensure_loaded()
        return self.df.sample(n=n, random_state=random_state)

    # ---------- Structure & summary ----------
    def shape(self) -> tuple[int, int]:
        """(rows, columns)."""
        self._ensure_loaded()
        return self.df.shape

    def info(self) -> str:
        """Return DataFrame info as a string."""
        self._ensure_loaded()
        buf = []
        self.df.info(buf=buf)
        return "\n".join(buf)

    def describe(self, include: Union[str, List[str]] = "all") -> pd.DataFrame:
        """Descriptive stats; include='all' to summarize mixed dtypes."""
        self._ensure_loaded()
        return self.df.describe(include=include)

    def missing_summary(self) -> pd.Series:
        """Count missing values per column."""
        self._ensure_loaded()
        return self.df.isna().sum().sort_values(ascending=False)

    # ---------- Column/row helpers ----------
    def columns(self) -> List[str]:
        """Alias for header()."""
        return self.header()

    def get(self, col: str) -> pd.Series:
        """Return a column by name."""
        self._ensure_loaded()
        return self.df[col]

    def value_counts(self, col: str, dropna: bool = True) -> pd.Series:
        """Value counts for a column."""
        self._ensure_loaded()
        return self.df[col].value_counts(dropna=dropna)

    def select(self, cols: List[str]) -> pd.DataFrame:
        """Select a subset of columns."""
        self._ensure_loaded()
        return self.df[cols]

    # ---------- Internal ----------
    def _ensure_loaded(self):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call .load() first.")

# ---------------- Example ----------------
if __name__ == "__main__":
    dr = DataReader("training_data_clean.csv")
    print("Columns:", dr.header())
    print("Shape:", dr.shape())
    print(dr.show(5))
    print(dr.describe())
    print("Missing:\n", dr.missing_summary())
