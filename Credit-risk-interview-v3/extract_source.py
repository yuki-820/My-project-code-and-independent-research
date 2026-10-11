"""Download the author's pinned PBIX and export its five model tables. No synthetic generation."""
from pathlib import Path
import hashlib,json,urllib.request,sys
from pbixray import PBIXRay
R=Path(__file__).resolve().parent
m=json.loads((R/'data/source_manifest.json').read_text(encoding='utf-8'))
p=R/'data/source.pbix'
if not p.exists():
    print('Downloading pinned public PBIX ...')
    req=urllib.request.Request(m['pbix_url'],headers={'User-Agent':'CreditCheck-study'})
    with urllib.request.urlopen(req,timeout=90) as resp: p.write_bytes(resp.read())
if hashlib.sha256(p.read_bytes()).hexdigest()!=m['pbix_sha256']:
    sys.exit('PBIX hash mismatch. Do not use this file. Remove it and retry the pinned URL.')
model=PBIXRay(str(p)); (R/'data/original').mkdir(exist_ok=True)
for t,expected in m['tables'].items():
    df=model.get_table(t)
    assert len(df)==expected['rows'] and list(df.columns)==expected['columns'],t
    out=R/'data/original'/f'{t}.csv';df.to_csv(out,index=False)
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    if digest!=expected['csv_sha256']:
        sys.exit(f'{t}: serialization hash mismatch. Check pinned dependencies; no manifest auto-overwrite.')
    print(t,len(df),'verified')
print('Ready: python run.py')
