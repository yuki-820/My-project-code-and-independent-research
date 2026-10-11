"""Three interview SQL fixtures, explicitly constructed for teaching, not UCI/CreditCheck data."""
import sqlite3,sys,math
from pathlib import Path
R=Path(__file__).resolve().parent
def connection():
 c=sqlite3.connect(':memory:')
 c.executescript('''CREATE TABLE videos(video_id TEXT,duration_sec INTEGER);
 CREATE TABLE plays(event_id TEXT,video_id TEXT,watch_sec INTEGER);
 CREATE TABLE logins(user_id TEXT,login_date TEXT);
 CREATE TABLE orders(order_id TEXT,paid_date TEXT,amount REAL,status TEXT);
 INSERT INTO videos VALUES('v1',10),('v2',20),('v3',0),('v4',15);
 INSERT INTO plays VALUES('e1','v1',10),('e1','v1',10),('e2','v1',5),('e3','v2',21),('e4','v3',1),('e5','v9',10),('e6','v1',NULL);
 INSERT INTO logins VALUES('u1','2026-10-01'),('u1','2026-10-01'),('u1','2026-10-02'),('u1','2026-10-03'),('u1','2026-10-05'),('u2','2026-10-01'),('u2','2026-10-03'),('u3','2026-10-02'),('u3','2026-10-03'),('u3','2026-10-04');
 INSERT INTO orders VALUES('o1','2026-10-01',100,'paid'),('o2','2026-10-03',200,'paid'),('o3','2026-10-03',50,'paid'),('o4','2026-10-08',300,'paid'),('o5','2026-10-10',400,'paid'),('o6','2026-10-04',999,'cancelled');''')
 return c
EXPECTED={
 'I01':[('v1',2,1,50.0),('v2',1,1,100.0),('v4',0,0,None)],
 'I02':[('u1','2026-10-01','2026-10-03',3),('u3','2026-10-02','2026-10-04',3)],
 'I03':[('2026-10-01',100,100),('2026-10-02',0,100),('2026-10-03',250,350),('2026-10-04',0,350),('2026-10-05',0,350),('2026-10-06',0,350),('2026-10-07',0,350),('2026-10-08',300,550),('2026-10-09',0,550),('2026-10-10',400,700)]}
def main():
 c=connection();ids=[sys.argv[1].upper()] if len(sys.argv)>1 else list(EXPECTED)
 for qid in ids:
  folder='interview_exercises' if len(sys.argv)>1 else 'interview_answers'
  p=R/'sql'/folder/f'{qid}.sql';got=c.execute(p.read_text(encoding='utf-8')).fetchall()
  assert got==EXPECTED[qid],f'{qid}: got={got}; expected={EXPECTED[qid]}'
  print('PASS',qid,got)
if __name__=='__main__':main()
