import unittest,sys,itertools,random,sqlite3
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from algorithms import three_sum,quick_sort,max_below
class FixtureTests(unittest.TestCase):
 def test_source_hashes(self):
  manifest=json.loads((R/'data/source_manifest.json').read_text())
  for name,m in manifest['tables'].items():
   self.assertEqual(hashlib.sha256((R/f'data/original/{name}.csv').read_bytes()).hexdigest(),m['csv_sha256'])
 def test_three_sql_fixtures(self):
  from interview_cases import connection,EXPECTED
  c=connection()
  for q,expected in EXPECTED.items():
   self.assertEqual(c.execute((R/f'sql/interview_answers/{q}.sql').read_text()).fetchall(),expected)
 def test_real_source_hash(self):
  m=json.loads((R/'real_project/data/source.json').read_text())
  self.assertEqual(hashlib.sha256((R/'real_project/data/clients.csv').read_bytes()).hexdigest(),m['csv_sha256'])
class AlgorithmTests(unittest.TestCase):
 def test_three_sum_bruteforce(self):
  rnd=random.Random(23)
  for _ in range(120):
   a=[rnd.randrange(-5,6) for _ in range(rnd.randrange(0,10))]
   expected=sorted(set(tuple(sorted(c)) for c in itertools.combinations(a,3) if sum(c)==0))
   self.assertEqual(sorted(map(tuple,three_sum(a))),expected)
 def test_sort(self):
  rnd=random.Random(24)
  for _ in range(100):
   a=[rnd.randrange(-10,11) for _ in range(rnd.randrange(0,35))]
   self.assertEqual(quick_sort(a),sorted(a))
 def test_max_below_bruteforce(self):
  for ds in [[0],[2,4,9],[0,1],[0,2,9],[1,2],[9],[]]:
   for n in range(0,350):
    valid=[x for x in range(n) if set(str(x))<=set(map(str,ds))]
    self.assertEqual(max_below(n,ds),max(valid) if valid else None,(n,ds))
  self.assertEqual(max_below(23121,[2,4,9]),22999)

class PandasChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import pandas_lab
  cls.p=pandas_lab;cls.d=pandas_lab.load();cls.c=sqlite3.connect(R/'data/creditcheck.db')
 def test_groupby_sql(self):
  got=self.p.P02(self.d)
  rows=self.c.execute("SELECT branch_id,COUNT(*),SUM(approval_status='Approved') FROM loan_applications GROUP BY branch_id ORDER BY branch_id").fetchall()
  self.assertEqual(got[['branch_id','applications','approved']].values.tolist(),[list(x) for x in rows])
 def test_merge_and_pivot(self):
  self.assertEqual(len(self.p.P03(self.d)),6112)
  self.assertEqual(self.p.P07(self.d).to_numpy().sum(),7368293000)
  out=self.p.P04(self.d);self.assertAlmostEqual(out.total_paid.sum(),4498935568.37,places=2)
 def test_latest_and_transform(self):
  got=self.p.P05(self.d)
  self.assertTrue(got.customer_id.is_unique)
  sql=(R/'sql/answers/Q17.sql').read_text(encoding='utf-8')
  self.assertEqual(got.application_id.tolist(),[x[1] for x in self.c.execute(sql)])
  s=self.p.P06(self.d).groupby('branch_id')['share'].sum()
  self.assertTrue(((s-1).abs()<1e-10).all())
 def test_scorebands_against_sql(self):
  got=self.p.P08(self.d);rows=self.c.execute((R/'sql/answers/Q08.sql').read_text(encoding='utf-8')).fetchall()
  self.assertEqual(got.astype({'band':'str'}).values.tolist(),[list(x) for x in rows])
if __name__=='__main__':unittest.main()
