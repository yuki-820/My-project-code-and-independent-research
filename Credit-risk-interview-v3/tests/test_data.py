import unittest,sqlite3,csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
class SourceChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.c=sqlite3.connect(R/'data/creditcheck.db')
 def test_counts_and_unique_keys(self):
  for t,k,n in [('branches','branch_id',25),('customers','customer_id',5000),('loan_applications','application_id',7500),('loans','loan_id',6112),('repayments','payment_id',153170)]:
   self.assertEqual(self.c.execute(f'SELECT COUNT(*),COUNT(DISTINCT {k}) FROM {t}').fetchone(),(n,n))
 def test_foreign_keys(self):
  for a,k,b in [('loan_applications','customer_id','customers'),('loan_applications','branch_id','branches'),('loans','application_id','loan_applications'),('loans','customer_id','customers'),('loans','branch_id','branches'),('repayments','loan_id','loans')]:
   self.assertEqual(self.c.execute(f'SELECT COUNT(*) FROM {a} a LEFT JOIN {b} b ON a.{k}=b.{k} WHERE b.{k} IS NULL').fetchone()[0],0)
 def test_python_independent_totals(self):
  with (R/'data/original/loans.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
  self.assertEqual(sum(int(r['loan_amount']) for r in rows),7368293000)
  self.assertEqual(sum(r['status']=='defaulted' for r in rows),378)
 def test_approval_mapping(self):
  self.assertEqual(self.c.execute("SELECT COUNT(*) FROM loans l JOIN loan_applications a USING(application_id) WHERE a.approval_status<>'Approved' OR a.customer_id<>l.customer_id OR a.branch_id<>l.branch_id").fetchone()[0],0)
 def test_bucket_matches_embedded_dax(self):
  self.assertEqual(self.c.execute("""SELECT COUNT(*) FROM loans l JOIN
   (SELECT loan_id,MAX(CASE WHEN payment_status='missed' THEN dpd ELSE 0 END) d FROM repayments GROUP BY loan_id) r USING(loan_id)
   WHERE l."Current DPD"<>r.d""").fetchone()[0],0)
 def test_documented_status_exception(self):
  self.assertEqual(self.c.execute("SELECT COUNT(*) FROM repayments WHERE payment_status='on_time' AND dpd>0").fetchone()[0],42320)
if __name__=='__main__':unittest.main()
