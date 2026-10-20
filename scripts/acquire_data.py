"""Download verified original source outside the repository; never commit it."""

import argparse
import hashlib
from pathlib import Path

import requests

URL = "https://data.mendeley.com/public-files/datasets/wm6w2fvkfj/files/19bc39c8-479d-4dfb-9bca-787022ca3a8b/file_downloaded"
SHA = "903f30aa008c478dd6451550b53c2bf937653f28763ea288e43f5e99c536cbdf"
p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
args = p.parse_args()
out = Path(args.output).resolve()
root = Path(__file__).resolve().parents[1]
if out.is_relative_to(root):
    raise SystemExit("Save the sensitive original outside the code repository.")
response = requests.get(URL, timeout=90)
response.raise_for_status()
payload = response.content
if hashlib.sha256(payload).hexdigest() != SHA:
    raise SystemExit("Provider file checksum differs; stop and audit the new source.")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(payload)
out.chmod(0o600)
print(f"Verified source saved outside repository: {len(payload)} bytes. Do not publish raw records.")
