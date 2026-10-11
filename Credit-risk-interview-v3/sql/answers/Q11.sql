-- 地区风险分布
-- 检查branch_id唯一，维表重复会放大分母和金额。
SELECT b.region,COUNT(*) AS loans,SUM(l.loan_amount) AS original_principal,
 SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 ROUND(100.0*SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END)/COUNT(*),2) AS default_rate_pct
FROM loans l JOIN branches b USING(branch_id) GROUP BY b.region ORDER BY original_principal DESC,b.region;
