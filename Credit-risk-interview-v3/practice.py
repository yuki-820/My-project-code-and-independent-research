"""Run an exercise and compare rows/columns with the reference on the lab dataset."""
import sys,sqlite3,math
from pathlib import Path
R=Path(__file__).resolve().parent
qid=sys.argv[1].upper() if len(sys.argv)>1 else 'Q01'
if qid not in [f'Q{x:02}' for x in range(1,21)]:raise SystemExit('Use Q01..Q20')
c=sqlite3.connect(f'file:{R / "data/creditcheck.db"}?mode=ro',uri=True);c.create_function('LN',1,math.log)
def fetch(p):
 cur=c.execute(p.read_text(encoding='utf-8'));return [x[0] for x in cur.description],cur.fetchall()
def close(a,b):
 if isinstance(a,(int,float)) and isinstance(b,(int,float)):return math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-6)
 return a==b
try:
 cols,got=fetch(R/f'sql/exercises/{qid}.sql');ec,expected=fetch(R/f'sql/answers/{qid}.sql')
 ok=cols==ec and len(got)==len(expected) and all(len(a)==len(b) and all(close(x,y) for x,y in zip(a,b)) for a,b in zip(got,expected))
 print('PASS' if ok else 'RETRY',qid, 'got rows:',len(got),'expected rows:',len(expected))
 print('expected columns:',ec)
 if not ok:print('Your first rows:',got[:5]);print('Expected first rows:',expected[:5])
 print('通过只代表此数据集结果一致；还要口述粒度、分母、NULL、时间边界。')
except sqlite3.Error as e:print('SQL ERROR:',e);sys.exit(1)
