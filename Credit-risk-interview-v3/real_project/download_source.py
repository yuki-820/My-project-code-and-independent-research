"""Fetch official UCI bytes, verify hash, convert original .xls without dropping records."""
from pathlib import Path
import json,urllib.request,hashlib,zipfile,io
import pandas as pd
R=Path(__file__).resolve().parent
m=json.loads((R/'data/source.json').read_text())
with urllib.request.urlopen(m['url'],timeout=90) as response:raw=response.read()
assert hashlib.sha256(raw).hexdigest()==m['zip_sha256'],'Source ZIP changed: inspect, do not silently accept'
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    xls=z.read(next(n for n in z.namelist() if n.endswith('.xls')))
assert hashlib.sha256(xls).hexdigest()==m['xls_sha256']
(R/'data/uci_original.xls').write_bytes(xls)
d=pd.read_excel(io.BytesIO(xls),header=1)
assert len(d)==m['rows']
p=R/'data/clients.csv';d.to_csv(p,index=False)
assert hashlib.sha256(p.read_bytes()).hexdigest()==m['csv_sha256'],'CSV serialization differs: check versions'
print('UCI official source and 30,000 rows verified. Run python real_project/run_real.py from project root.')
