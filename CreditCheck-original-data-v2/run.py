"""Build SQLite from the author's CSV exports and execute all 20 analyses. stdlib only."""
import csv,hashlib,json,sqlite3
from pathlib import Path
R=Path(__file__).resolve().parent
m=json.loads((R/'data/source_manifest.json').read_text(encoding='utf-8'))
def quoted(x):return '"'+x.replace('"','""')+'"'
ints={'branch_id','customer_id','application_id','loan_id','payment_id','age','credit_score','requested_amount','loan_amount','tenure_months','dpd','Current DPD'}
floats={'income','risk_score','interest_rate','amount_due','amount_paid'}
# Verify all inputs before touching the generated DB.
for t,meta in m['tables'].items():
 p=R/'data/original'/f'{t}.csv'
 if not p.exists():raise SystemExit('Missing author CSV exports. Run: pip install -r requirements-extract.txt ; python extract_source.py')
 if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['csv_sha256']:raise SystemExit(f'{t}: source CSV hash mismatch')
p=R/'data/creditcheck.db'
if p.exists():p.unlink()
c=sqlite3.connect(p)
for t,meta in m['tables'].items():
 cols=meta['columns']; types=['INTEGER' if x in ints else 'REAL' if x in floats else 'TEXT' for x in cols]
 c.execute(f'CREATE TABLE {quoted(t)} ('+', '.join(quoted(x)+' '+ty for x,ty in zip(cols,types))+')')
 with (R/'data/original'/f'{t}.csv').open(encoding='utf-8',newline='') as f:
  rd=csv.DictReader(f);rows=[]
  for row in rd:
   rows.append(tuple(None if row[x]=='' else int(row[x]) if x in ints else float(row[x]) if x in floats else row[x] for x in cols))
 c.executemany(f'INSERT INTO {quoted(t)} VALUES ('+','.join('?' for x in cols)+')',rows)
 assert len(rows)==meta['rows'],t
for table,columns in [('loans','loan_id'),('loans','customer_id'),('repayments','loan_id'),('customers','customer_id'),('branches','branch_id'),('loan_applications','application_id')]:
 c.execute(f'CREATE INDEX idx_{table}_{columns} ON {table}({columns})')
c.commit();(R/'reports').mkdir(exist_ok=True)
answers={}
for sql in sorted((R/'sql/answers').glob('Q*.sql')):
 cur=c.execute(sql.read_text(encoding='utf-8'));columns=[x[0] for x in cur.description];rows=cur.fetchall()
 with (R/'reports'/f'{sql.stem}.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(columns);w.writerows(rows)
 answers[sql.stem]={'columns':columns,'rows':rows}
(R/'reports/answers.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2),encoding='utf-8')
print('Author source hashes verified. 5 tables loaded. 20 SQL queries exported.')
print('Loans / principal / active / defaulted / default rate:',answers['Q01']['rows'][0])
print('Applications / approved / approval rate:',answers['Q02']['rows'][0])
c.close()
