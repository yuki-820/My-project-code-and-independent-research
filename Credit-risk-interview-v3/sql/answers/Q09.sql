-- 收入分层
-- 收入周期和单位的业务定义未获确认，不擅自构造DTI。
WITH s AS (SELECT l.status,c.income,
 CASE WHEN income<30000 THEN '<30k' WHEN income<50000 THEN '30k-50k'
 WHEN income<75000 THEN '50k-75k' WHEN income<100000 THEN '75k-100k' ELSE '100k+' END AS band
 FROM loans l JOIN customers c USING(customer_id))
SELECT band,COUNT(*) AS loans,SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 ROUND(100.0*SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END)/COUNT(*),2) AS default_rate_pct
FROM s GROUP BY band ORDER BY MIN(income);
