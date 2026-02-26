import argparse
import os
import sys
import time
from html.parser import HTMLParser

import requests
from tqdm import tqdm

BASE_URL    = "https://physionet.org/files/chbmit/1.0.0"
CHUNK_SIZE  = 1024 * 1024   # 1 MB download chunks
ALL_PATIENTS = list(range(1, 25))


# ── HTML link parser ──────────────────────────────────────────────────────────

class _LinkParser(HTMLParser):
    """Extracts href values from an HTML directory listing page."""
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value and not value.startswith("?") and not value.startswith("/"):
                    self.links.append(value)


def _list_patient_files(patient_dir: str) -> list[str]:
    """
    Fetches the PhysioNet directory listing for one patient and returns
    all .edf and *summary.txt filenames.
    """
    url = f"{BASE_URL}/{patient_dir}/"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Could not fetch file list for {patient_dir}: {e}")
        return []

    parser = _LinkParser()
    parser.feed(resp.text)

    # Keep only EDF files and the summary text file
    wanted = [
        f for f in parser.links
        if f.endswith(".edf") or ("summary" in f.lower() and f.endswith(".txt"))
    ]
    return wanted


# ── Single-file downloader ─────────────────────────────────────────────────────

def _download_file(url: str, dest_path: str) -> bool:
    """
    Downloads one file from `url` to `dest_path` with a progress bar.
    Skips download if the file already exists and has a non-zero size.
    Returns True on success, False on failure.
    """
    # Skip if already downloaded
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  [SKIP] Already exists: {os.path.basename(dest_path)}")
        return True

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        return False

    total_bytes = int(resp.headers.get("content-length", 0))
    filename    = os.path.basename(dest_path)

    with (
        open(dest_path, "wb") as f,
        tqdm(
            total=total_bytes,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"  {filename[:35]:<35}",
            leave=False,
        ) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    return True


# ── Patient downloader ─────────────────────────────────────────────────────────

def download_patient(patient_id: int, output_dir: str) -> bool:
    """
    Downloads all .edf files and the summary .txt for one patient.
    Returns True if all files downloaded successfully.
    """
    pdir              = f"chb{patient_id:02d}"
    local_patient_dir = os.path.join(output_dir, pdir)
    os.makedirs(local_patient_dir, exist_ok=True)

    print(f"\n[{pdir}] Fetching file list from PhysioNet...")
    files = _list_patient_files(pdir)

    if not files:
        print(f"  [ERROR] No files found for {pdir}. Check your internet connection.")
        return False

    edf_files = [f for f in files if f.endswith(".edf")]
    txt_files = [f for f in files if f.endswith(".txt")]
    print(f"  Found {len(edf_files)} EDF file(s) + {len(txt_files)} summary file(s)")

    success = True
    for filename in sorted(files):
        url       = f"{BASE_URL}/{pdir}/{filename}"
        dest_path = os.path.join(local_patient_dir, filename)
        ok        = _download_file(url, dest_path)
        if not ok:
            success = False

    if success:
        print(f"  [{pdir}] Done → {local_patient_dir}")
    else:
        print(f"  [{pdir}] Completed with some errors — run again to retry failed files")

    return success


# ── Verification ───────────────────────────────────────────────────────────────

def verify_download(patient_id: int, output_dir: str) -> dict:
    """Checks what files are present for a patient and reports status."""
    pdir       = f"chb{patient_id:02d}"
    local_path = os.path.join(output_dir, pdir)

    if not os.path.exists(local_path):
        return {"patient": pdir, "edf_count": 0, "has_summary": False,
                "size_mb": 0.0, "status": "MISSING"}

    files         = os.listdir(local_path)
    edf_files     = [f for f in files if f.endswith(".edf")]
    summary_files = [f for f in files if "summary" in f.lower() and f.endswith(".txt")]
    total_bytes   = sum(
        os.path.getsize(os.path.join(local_path, f)) for f in files
    )

    status = "OK" if (edf_files and summary_files) else "INCOMPLETE"
    return {
        "patient":      pdir,
        "edf_count":    len(edf_files),
        "has_summary":  bool(summary_files),
        "size_mb":      round(total_bytes / 1e6, 1),
        "status":       status,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download CHB-MIT EEG dataset from PhysioNet (pure Python, no wget needed)"
    )
    parser.add_argument(
        "--output", type=str, default="data/chb-mit",
        help="Local directory to save the dataset (default: data/chb-mit)"
    )
    parser.add_argument(
        "--patients", type=int, nargs="+", default=ALL_PATIENTS,
        help="Patient IDs to download. Example: --patients 1 2 3"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only check what is already downloaded, do not download anything"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Verify only ──
    if args.verify_only:
        print("\n=== Download Verification ===")
        print(f"  {'Patient':<10} {'EDFs':>6} {'Summary':>8} {'Size (MB)':>10}  Status")
        print(f"  {'-'*50}")
        for pid in args.patients:
            info = verify_download(pid, args.output)
            print(
                f"  {info['patient']:<10} {info['edf_count']:>6} "
                f"{'Yes' if info['has_summary'] else 'No':>8} "
                f"{info['size_mb']:>10.1f}  [{info['status']}]"
            )
        return

    # ── Download ──
    print(f"\nTarget directory : {os.path.abspath(args.output)}")
    print(f"Patients to download: {[f'chb{p:02d}' for p in args.patients]}")
    print(f"This may take a while for large patients (~1–3 GB each).\n")

    t_start  = time.time()
    failures = []
    for pid in args.patients:
        ok = download_patient(pid, args.output)
        if not ok:
            failures.append(f"chb{pid:02d}")

    # ── Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"DOWNLOAD COMPLETE  ({elapsed/60:.1f} min)")
    print(f"{'='*55}")
    print(f"  {'Patient':<10} {'EDFs':>6} {'Summary':>8} {'Size (MB)':>10}  Status")
    print(f"  {'-'*50}")
    for pid in args.patients:
        info = verify_download(pid, args.output)
        print(
            f"  {info['patient']:<10} {info['edf_count']:>6} "
            f"{'Yes' if info['has_summary'] else 'No':>8} "
            f"{info['size_mb']:>10.1f}  [{info['status']}]"
        )

    if failures:
        print(f"\n  [!] Failed: {failures}  — run again to retry")
    else:
        print(f"\n  All patients downloaded successfully.")


if __name__ == "__main__":
    main()
