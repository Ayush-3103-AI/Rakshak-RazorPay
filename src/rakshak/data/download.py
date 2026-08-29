"""Resumable downloader and provenance manifest for external datasets (T-0015).

Raw data lands in `data/external/`, which `.gitignore` excludes; only the
`*.manifest.json` files are tracked. The manifest is what the README cites.

Licences were verified at source on 2026-08-29 — see `SOURCES` below and the survey in
`results/calibration_gap.md`. Only `online_retail_ii` is fetchable without credentials;
the Kaggle-hosted sets need an API token in `~/.kaggle/kaggle.json`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from rakshak.cli import base_parser
from rakshak.config import DATA_DIR

EXTERNAL_DIR: Path = DATA_DIR / "external"
"""Where raw downloads land. Git-ignored except `*.manifest.json`."""

_CHUNK_BYTES: int = 1 << 20
"""Read/write chunk for streaming downloads and hashing. Units: bytes."""


@dataclass(frozen=True)
class Source:
    """One external dataset and the terms it may be used under.

    Attributes:
        name: Short key; also the manifest filename stem.
        url: Direct download URL, or the Kaggle dataset slug when `kaggle` is True.
        filename: Local filename under `data/external/`.
        licence: Licence as stated at `licence_url`, verified at source.
        licence_url: Where the licence statement was read.
        source_page: Human landing page for the dataset.
        kaggle: True when the URL is a Kaggle dataset slug needing an API token.
        note: What this source can and cannot inform.
    """

    name: str
    url: str
    filename: str
    licence: str
    licence_url: str
    source_page: str
    kaggle: bool = False
    note: str = ""


SOURCES: dict[str, Source] = {
    "online_retail_ii": Source(
        name="online_retail_ii",
        url="https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        filename="online_retail_ii.zip",
        licence="CC BY 4.0",
        licence_url="https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        source_page="https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        note=(
            "Real transaction stream of ONE UK online gift-ware retailer, Dec 2009 - Dec 2011. "
            "Informs within-merchant amount dispersion, hour-of-day and weekday seasonality, "
            "daily volume and its over-dispersion, repeat-payer structure and the "
            "return/cancellation rate. Cannot inform chargeback rate, fraud prevalence, "
            "cross-category amount levels, or anything about latent risk states."
        ),
    ),
    "baf": Source(
        name="baf",
        url="sgpjesus/bank-account-fraud-dataset-neurips-2022",
        filename="baf.zip",
        licence="CC BY-NC-SA 4.0",
        licence_url=(
            "https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022"
        ),
        source_page="https://github.com/feedzai/bank-account-fraud",
        kaggle=True,
        note=(
            "Bank ACCOUNT-OPENING applications, not transactions: no amount, no timestamp, "
            "no payer, no merchant, no sequences. Informs NONE of the generator's marginals "
            "(ADR-0007); its use is decision-layer validation on a real label distribution "
            "with real temporal drift (T-0012). Non-commercial + share-alike: usable inside "
            "the git-ignored data/ directory, NOT vendorable into this repo."
        ),
    ),
}
"""Datasets this repo will fetch. Rejected candidates are recorded in the survey, not here."""


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed.

    Args:
        path: File to hash.

    Returns:
        Lower-case hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(
    name: str,
    path: Path,
    source_url: str,
    licence: str,
    licence_url: str,
    row_count: int | None = None,
    subsample_seed: int | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the provenance record for one downloaded file.

    Args:
        name: Dataset key.
        path: The downloaded file on disk.
        source_url: Where it came from.
        licence: Licence text as stated at `licence_url`.
        licence_url: Where the licence was read, at source.
        row_count: Rows in the normalised table, if known.
        subsample_seed: Seed used if the data was subsampled; None if it was not.
        extra: Additional key/values to merge in.

    Returns:
        A JSON-serialisable manifest dict. `retrieved_utc` is an ISO-8601 UTC instant.
    """
    manifest: dict[str, object] = {
        "name": name,
        "source_url": source_url,
        "licence": licence,
        "licence_url": licence_url,
        "retrieved_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "row_count": row_count,
        "subsampled": subsample_seed is not None,
        "subsample_seed": subsample_seed,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict[str, object], directory: Path = EXTERNAL_DIR) -> Path:
    """Write `<name>.manifest.json` beside the data and return its path.

    Args:
        manifest: Output of `file_manifest`.
        directory: Where the data lives.

    Returns:
        Path to the written manifest.
    """
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{manifest['name']}.manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def kaggle_auth(config_dir: Path | None = None, env: Mapping[str, str] | None = None) -> str | None:
    """Return an `Authorization` header value for Kaggle, or None if no credential is found.

    Kaggle issues two credential formats and both are still in circulation:

    * the **current** one, an opaque API token prefixed `KGAT_`, supplied either as
      `KAGGLE_API_TOKEN` or in `~/.kaggle/access_token`, sent as a **Bearer** token;
    * the **legacy** one, `~/.kaggle/kaggle.json` holding `{"username", "key"}`, sent as
      HTTP **Basic**.

    Bearer wins when both are present, because that is the format Kaggle currently mints.

    Args:
        config_dir: Credential directory. Defaults to `$KAGGLE_CONFIG_DIR` or `~/.kaggle`.
        env: Environment mapping to read. Defaults to `os.environ`. Injectable so the
            resolution order can be tested without touching the real home directory or
            the real environment.

    Returns:
        The header value, or None when no credential is configured.
    """
    env = os.environ if env is None else env
    directory = Path(config_dir) if config_dir is not None else Path(
        env.get("KAGGLE_CONFIG_DIR") or (Path.home() / ".kaggle")
    )

    token = (env.get("KAGGLE_API_TOKEN") or "").strip()
    if not token:
        access_token = directory / "access_token"
        if access_token.exists():
            token = access_token.read_text(encoding="utf-8").strip()
    if token:
        return f"Bearer {token}"

    creds = directory / "kaggle.json"
    if not creds.exists():
        return None
    payload = json.loads(creds.read_text(encoding="utf-8"))
    basic = f"{payload['username']}:{payload['key']}".encode()
    return "Basic " + base64.b64encode(basic).decode()


def fetch(source: Source, directory: Path = EXTERNAL_DIR) -> Path:
    """Download `source` to `directory`, resuming a partial file if one is present.

    A `.part` file accumulates bytes and is renamed on completion, so an interrupted run
    resumes with an HTTP Range request rather than starting over.

    Args:
        source: The dataset to fetch.
        directory: Destination directory.

    Returns:
        Path to the completed file.

    Raises:
        RuntimeError: If the source needs Kaggle credentials and none are configured.
    """
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / source.filename
    if dest.exists():
        return dest

    headers: dict[str, str] = {"User-Agent": "rakshak-t0015/1.0"}
    url = source.url
    if source.kaggle:
        auth = kaggle_auth()
        if auth is None:
            raise RuntimeError(
                f"{source.name} is hosted on Kaggle and needs an API token at "
                "$KAGGLE_API_TOKEN, ~/.kaggle/access_token, or legacy ~/.kaggle/kaggle.json. "
                "No credentials found; refusing to fabricate data."
            )
        headers["Authorization"] = auth
        url = f"https://www.kaggle.com/api/v1/datasets/download/{source.url}"

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have:
        headers["Range"] = f"bytes={have}-"

    with urlopen(Request(url, headers=headers)) as response:  # noqa: S310
        mode = "ab" if have and response.status == 206 else "wb"
        with part.open(mode) as handle:
            for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                handle.write(chunk)
    part.replace(dest)
    return dest


def normalise_online_retail_ii(zip_path: Path) -> pd.DataFrame:
    """Read the UCI workbook and return one row per *invoice* in the canonical schema.

    A row in the source file is a line item; a row here is a payment, which is the unit
    the Rakshak generator emits. Amount is the invoice total in GBP (a positive
    magnitude; `is_refund` carries the direction, matching the generator's convention).

    Args:
        zip_path: The downloaded `online+retail+ii.zip`.

    Returns:
        DataFrame with columns `timestamp` (datetime64[ns]), `amount` (float, GBP),
        `payer_id` (str, empty when the source had no customer id) and `is_refund` (bool).
    """
    with zipfile.ZipFile(zip_path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as handle:
            sheets = pd.read_excel(handle, sheet_name=None, engine="openpyxl")

    frames = []
    for frame in sheets.values():
        frame = frame.rename(
            columns={"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID"}
        )
        frame["line_value"] = frame["Quantity"] * frame["Price"]
        frames.append(frame[["Invoice", "InvoiceDate", "Customer ID", "line_value"]])
    lines = pd.concat(frames, ignore_index=True)

    lines["Invoice"] = lines["Invoice"].astype(str)
    grouped = lines.groupby("Invoice", sort=True).agg(
        timestamp=("InvoiceDate", "min"),
        amount=("line_value", "sum"),
        payer_id=("Customer ID", "first"),
    )
    out = grouped.reset_index()
    out["is_refund"] = out["Invoice"].str.upper().str.startswith("C")
    out["amount"] = out["amount"].abs()
    out["payer_id"] = out["payer_id"].apply(lambda v: "" if pd.isna(v) else f"C{int(v)}")
    out = out[out["amount"] > 0.0]
    return (
        out[["timestamp", "amount", "payer_id", "is_refund"]]
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    """CLI: fetch a dataset, normalise it and write its manifest."""
    parser = base_parser("Download an external dataset and write its provenance manifest.")
    parser.add_argument("--dataset", default="online_retail_ii", choices=sorted(SOURCES))
    args = parser.parse_args()

    source = SOURCES[args.dataset]
    path = fetch(source)
    row_count: int | None = None
    if source.name == "baf":
        # ADR-0007 requires row count in the manifest. Counted from the Base variant
        # without extracting it; the other five bias variants are not used.
        with zipfile.ZipFile(path) as archive, archive.open("Base.csv") as handle:
            row_count = sum(1 for _ in handle) - 1  # minus the header
    if source.name == "online_retail_ii":
        frame = normalise_online_retail_ii(path)
        frame.to_parquet(EXTERNAL_DIR / "online_retail_ii.parquet", index=False)
        row_count = len(frame)
    manifest = file_manifest(
        name=source.name,
        path=path,
        source_url=source.url,
        licence=source.licence,
        licence_url=source.licence_url,
        row_count=row_count,
        extra={"source_page": source.source_page, "note": source.note},
    )
    print(f"wrote {write_manifest(manifest)}")


if __name__ == "__main__":
    main()
