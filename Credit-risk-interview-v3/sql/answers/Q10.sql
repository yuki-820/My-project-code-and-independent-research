-- 就业类型分层
-- 相关性不能证明因果；合成机制也可能预先制造这种相关。
SELECT c.employment_type,COUNT(*) AS loans,
 SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 ROUND(100.0*SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END)/COUNT(*),2) AS default_rate_pct
FROM loans l JOIN customers c USING(customer_id) GROUP BY c.employment_type ORDER BY default_rate_pct DESC,c.employment_type;
