-- 支行风险排名
-- 按未四舍五入率排名，减少假并列；小支行须结合样本量。原作者按四舍五入率排名。
WITH s AS (SELECT b.branch_id,b.branch_name,COUNT(*) AS loans,
 SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 100.0*SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END)/COUNT(*) AS rate
 FROM loans l JOIN branches b USING(branch_id) GROUP BY b.branch_id,b.branch_name)
SELECT branch_id,branch_name,loans,defaults,ROUND(rate,2) AS default_rate_pct,
 RANK() OVER(ORDER BY rate DESC) AS risk_rank FROM s ORDER BY risk_rank,branch_id;
