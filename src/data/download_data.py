"""Download the original CSV from the official UCI archive, never a mirror."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
URL = 'https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip'
CSV = ROOT / 'data/raw/energydata_complete.csv'

def download():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    if not CSV.exists():
        with urlopen(URL, timeout=90) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            CSV.write_bytes(archive.read('energydata_complete.csv'))
        provenance = {'url': URL, 'doi': '10.24432/C5VC8G', 'license': 'CC BY 4.0',
                      'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                      'method': 'urllib HTTPS official UCI ZIP; extract named CSV',
                      'sha256': hashlib.sha256(CSV.read_bytes()).hexdigest()}
        (CSV.parent / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return CSV

if __name__ == '__main__':
    print(download())
