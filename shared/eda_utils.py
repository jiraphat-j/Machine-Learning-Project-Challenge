from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image


TASKS = {
    1: ("01_binary_classification", "is_cheater", "task1_anti_cheat"),
    2: ("02_multiclass_classification", "segment", "task2_segmentation"),
    3: ("03_regression", "spending_30d", "task3_spending"),
    4: ("04_image_classification/task4_image_classification", "class", "task4_images"),
    5: ("05_anomaly_detection", "is_anomaly", "task5_anomaly"),
}


def find_root():
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "sample_submission.csv").exists():
            return path
    raise FileNotFoundError("Run this notebook inside the project folder.")


def setup_plot():
    font = Path("C:/Windows/Fonts/tahoma.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Tahoma"
    plt.rcParams.update({"figure.figsize": (9, 4), "axes.grid": True,
                         "axes.axisbelow": True, "axes.unicode_minus": False})
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.max_rows", 15)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def load_task(root, task):
    folder, target, _ = TASKS[task]
    base = root / folder
    train = pd.read_csv(base / "train.csv") if (base / "train.csv").exists() else None
    test = pd.read_csv(base / "test.csv")
    return train, test, target, base


def quality_table(frame):
    result = pd.DataFrame({"dtype": frame.dtypes.astype(str),
                           "missing": frame.isna().sum(),
                           "missing_pct": frame.isna().mean() * 100,
                           "unique": frame.nunique()})
    return result.sort_values("missing_pct", ascending=False)


def outlier_table(frame):
    numeric = frame.select_dtypes("number")
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3 - q1
    flags = (numeric.lt(q1 - 1.5 * iqr) | numeric.gt(q3 + 1.5 * iqr)) & numeric.notna()
    result = pd.DataFrame({"iqr_outliers": flags.sum(),
                           "observed": numeric.notna().sum()})
    result["outlier_pct"] = result.iqr_outliers / result.observed * 100
    return result.sort_values("outlier_pct", ascending=False)


def bounds_table(frame, task):
    bounds = {}
    if task == 1:
        bounds = {"headshot_percentage": (0, 100), "win_rate": (0, 100),
                  "accuracy_score": (0, 100), "night_play_ratio": (0, 1),
                  "reaction_time_ms": (0, np.inf), "account_age_days": (0, np.inf)}
    elif task == 2:
        bounds = {"play_frequency": (0, 7), "peak_concurrent_hours": (0, 24),
                  "team_play_percentage": (0, 100), "total_spending_thb": (0, np.inf)}
    elif task == 3:
        bounds = {"guild_membership": (0, 1), "is_premium_member": (0, 1),
                  "segment": (0, 3), "prev_month_spending": (0, np.inf),
                  "historical_spending": (0, np.inf), "spending_30d": (0, np.inf)}
    elif task == 5:
        for name in frame:
            if name.startswith("login_lat_"):
                bounds[name] = (-90, 90)
            elif name.startswith("login_lon_"):
                bounds[name] = (-180, 180)
            elif name.startswith("ip_hash_entropy_"):
                bounds[name] = (0, 1)
            elif name.startswith(("login_count_", "location_changes_", "device_count_",
                                  "refund_count_", "guild_changes_", "friend_adds_",
                                  "friend_removes_", "purchase_frequency_",
                                  "purchase_amount_mean_", "purchase_amount_std_",
                                  "currency_transfers_")):
                bounds[name] = (0, np.inf)
        for name in ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]:
            bounds[name] = (0, 1)
    rows = []
    for name, (low, high) in bounds.items():
        if name in frame:
            outside = frame[name].notna() & ~frame[name].between(low, high)
            rows.append({"feature": name, "expected_min": low, "expected_max": high,
                         "outside_range": int(outside.sum())})
    return pd.DataFrame(rows)


def association(frame, target, regression=False):
    numeric = frame.select_dtypes("number").drop(columns=[target], errors="ignore")
    valid = frame[target].notna()
    numeric, y = numeric.loc[valid], frame.loc[valid, target]
    if regression:
        return numeric.corrwith(y, method="spearman").abs().sort_values(ascending=False)
    scores = {}
    for name in numeric:
        data = pd.DataFrame({"x": numeric[name], "y": y}).dropna()
        mean = data.x.mean()
        groups = data.groupby("y").x.agg(["count", "mean"])
        total = ((data.x - mean) ** 2).sum()
        scores[name] = float((groups["count"] * (groups["mean"] - mean) ** 2).sum() / total) if total else 0
    return pd.Series(scores, name="eta_squared").sort_values(ascending=False)


def strongest_pairs(frame, limit=10):
    corr = frame.select_dtypes("number").corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool), 1)
    return corr.where(mask).stack().sort_values(key=lambda x: x.abs(), ascending=False).head(limit)


def period_changes(frame):
    rows = []
    for name in frame:
        if name.endswith("_1") and name[:-2] + "_4" in frame:
            last = name[:-2] + "_4"
            delta = frame[last] - frame[name]
            rows.append({"feature": name[:-2], "paired_rows": int(delta.notna().sum()),
                         "median_delta": delta.median(), "p95_abs_delta": delta.abs().quantile(.95),
                         "p99_abs_delta": delta.abs().quantile(.99)})
    return pd.DataFrame(rows).set_index("feature")


def image_inventory(base, frame, split):
    rows = []
    for image_id in frame.id:
        path = base / split / f"{image_id}.jpg"
        row = {"id": image_id, "exists": path.exists(), "readable": False,
               "width": np.nan, "height": np.nan, "mode": "", "error": "", "sha256": ""}
        if row["exists"]:
            try:
                with Image.open(path) as image:
                    image.load()
                    row.update(readable=True, width=image.width, height=image.height, mode=image.mode)
                row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError) as error:
                row["error"] = type(error).__name__
        rows.append(row)
    return pd.DataFrame(rows)
