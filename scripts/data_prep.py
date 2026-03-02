from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
# Feature sets
# -------------------------------------------------------------------

CLUSTER_FEATURES: List[str] = [
    "notes_per_second_mean",
    "chord_ratio",
    "sustain_ratio",
    "spectral_centroid_mean",
    "rms_energy_mean",
]


# -------------------------------------------------------------------
# Generic utilities
# -------------------------------------------------------------------

def clean_df(df: pd.DataFrame, cols: Sequence[str], extra_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Select columns, replace inf with NaN, drop rows with missing values."""
    cols = list(cols)
    if extra_cols:
        cols = list(dict.fromkeys(list(extra_cols) + cols))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")
    return df.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna()


def standardize(X: np.ndarray) -> np.ndarray:
    """Z-score standardize features."""
    return StandardScaler().fit_transform(X)


def get_cluster_matrix(
    features_df: pd.DataFrame,
    feature_set: Sequence[str] = CLUSTER_FEATURES,
) -> np.ndarray:
    """
    Converts the features df into a standardized numpy matrix aligned to its rows.
    """
    feature_set = list(feature_set)
    missing = [c for c in feature_set if c not in features_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in features_df: {missing}")
    X = features_df.loc[:, feature_set].to_numpy()
    return standardize(X)


def select_feature_df(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    id_cols: Sequence[str] = ("song_id", "instrument_id", "difficulty_id"),
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Selects a dataframe containing id_cols plus feature_cols.
    Replaces inf with NaN, optionally drops rows with missing values.

    Use this to replace: clean_df(df_features, some_feature_list)
    """
    feature_cols = list(feature_cols)
    id_cols = list(id_cols) if id_cols else []
    cols = list(dict.fromkeys(id_cols + feature_cols))

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    out = df.loc[:, cols].replace([np.inf, -np.inf], np.nan)
    if dropna:
        out = out.dropna()
    return out


# -------------------------------------------------------------------
# Row cleaning logic (no column pruning)
# -------------------------------------------------------------------

@dataclass
class CleaningReport:
    removed_inconsistent_duration_records: int
    removed_low_note_records: int
    removed_low_note_songs: int
    set_near_zero_fret_entropy_to_zero: int


def keep_highest_difficulty_per_song_instrument(full: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the highest difficulty record within each (song_id, instrument_id) group.
    """
    key_cols = ["song_id", "instrument_id", "difficulty_id"]
    missing = [c for c in key_cols if c not in full.columns]
    if missing:
        raise KeyError(f"Expected columns missing for highest difficulty step: {missing}")

    highest = (
        full.loc[:, key_cols]
        .groupby(["song_id", "instrument_id"])
        .agg(max)
        .reset_index()
    )
    merged = pd.merge(highest, full, how="inner", on=key_cols)
    return merged


def keep_highest_difficulty_only(df: pd.DataFrame, drop_difficulty_id: bool = False) -> pd.DataFrame:
    """
    Wrapper around keep_highest_difficulty_per_song_instrument.
    Collapses to one row per song_id, instrument_id by keeping max difficulty_id.
    """
    out = keep_highest_difficulty_per_song_instrument(df)
    if drop_difficulty_id and "difficulty_id" in out.columns:
        out = out.drop(columns=["difficulty_id"])
    return out


def remove_inconsistent_durations(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Removes records for songs where duration differs across instruments by
    dropping the least frequent duration option for that song.
    """
    needed = ["song_id", "instrument_id", "duration_sec"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns missing for duration check: {missing}")

    tmp = df.loc[:, needed].drop_duplicates()
    dur_list = tmp.groupby("song_id")["duration_sec"].agg(list).to_frame()
    dur_list["_len"] = dur_list["duration_sec"].apply(len)
    bad_songs = dur_list.index[dur_list["_len"] > 1].tolist()

    if not bad_songs:
        return df, 0

    bad = df.loc[df["song_id"].isin(bad_songs), ["song_id", "duration_sec"]].copy()
    bad = bad.reset_index(names="record_index")

    counts = (
        bad.groupby(["song_id", "duration_sec"])
           .agg(counts=("duration_sec", "size"), indices=("record_index", list))
           .reset_index()
    )

    worst = counts.sort_values(["song_id", "counts"]).explode("indices")
    all_but_last = worst.duplicated(subset="song_id", keep="last")
    worst_indices = worst.loc[all_but_last, "indices"].astype(int).tolist()

    prev = len(df)
    df2 = df.loc[~df.index.isin(worst_indices)].copy()
    removed = prev - len(df2)
    return df2, int(removed)


def remove_suspicious_low_note_songs(df: pd.DataFrame, n_notes_threshold: int = 20) -> Tuple[pd.DataFrame, int, int]:
    """
    Drops entire songs if any instrument record in that song has very low note count,
    which likely indicates a parsing or chart issue.
    """
    needed = ["song_id", "n_notes"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns missing for low note filter: {missing}")

    low = df.loc[df["n_notes"] < int(n_notes_threshold), ["song_id"]]
    songs_to_remove = low["song_id"].unique().tolist()

    prev = len(df)
    df2 = df.loc[~df["song_id"].isin(songs_to_remove)].copy()
    removed_records = prev - len(df2)

    return df2, int(removed_records), int(len(songs_to_remove))


def set_near_zero_fret_entropy_to_zero(df: pd.DataFrame, cutoff: float = 1e-5) -> Tuple[pd.DataFrame, int]:
    """
    Sets extremely small positive fret_entropy values to 0 to remove numerical noise.

    If fret_entropy is not present, it is a no op.
    """
    if "fret_entropy" not in df.columns:
        return df, 0

    df2 = df.copy()
    mask = (df2["fret_entropy"] > 0) & (df2["fret_entropy"] < float(cutoff))
    n = int(mask.sum())
    df2.loc[mask, "fret_entropy"] = 0.0
    return df2, n


def clean_rows_for_eda(
    df: pd.DataFrame,
    n_notes_threshold: int = 20,
    fret_entropy_cutoff: float = 1e-5,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Row cleanup intended for exploratory work.
    Preserves instrument_id and difficulty_id, keeps all columns.
    Does NOT collapse to highest difficulty.
    """
    df2 = df.copy()

    df2, removed_dur = remove_inconsistent_durations(df2)

    df2, removed_low_note_records, removed_low_note_songs = remove_suspicious_low_note_songs(
        df2, n_notes_threshold=int(n_notes_threshold)
    )

    df2, n_fret_fixed = set_near_zero_fret_entropy_to_zero(df2, cutoff=float(fret_entropy_cutoff))

    report = CleaningReport(
        removed_inconsistent_duration_records=int(removed_dur),
        removed_low_note_records=int(removed_low_note_records),
        removed_low_note_songs=int(removed_low_note_songs),
        set_near_zero_fret_entropy_to_zero=int(n_fret_fixed),
    )
    return df2, report


# -------------------------------------------------------------------
# Instrument isolation (filtering)
# -------------------------------------------------------------------

def filter_instruments(
    df: pd.DataFrame,
    instruments: Sequence[int],
    instrument_col: str = "instrument_id",
) -> pd.DataFrame:
    """
    Filters df to only the specified instrument ids.
    """
    if instrument_col not in df.columns:
        raise KeyError(f"Expected column missing: {instrument_col}")
    inst = set(int(x) for x in instruments)
    return df.loc[df[instrument_col].astype(int).isin(inst)].copy()


def isolate_instrument(df: pd.DataFrame, instrument_id: int) -> pd.DataFrame:
    """
    Convenience wrapper for selecting a single instrument.
    """
    return filter_instruments(df, instruments=[int(instrument_id)])