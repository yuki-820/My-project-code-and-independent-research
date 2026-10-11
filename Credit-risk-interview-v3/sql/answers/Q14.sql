-- 支行审批排名
-- RANK并列跳号，DENSE_RANK不跳号，ROW_NUMBER每行唯一序号。
WITH s AS (SELECT b.branch_id,b.branch_name,COUNT(*) AS applications,
 SUM(CASE WHEN a.approval_status='Approved' THEN 1 ELSE 0 END) AS approved,
 100.0*SUM(CASE WHEN a.approval_status='Approved' THEN 1 ELSE 0 END)/COUNT(*) AS rate
 FROM loan_applications a JOIN branches b USING(branch_id) GROUP BY b.branch_id,b.branch_name)
SELECT branch_id,branch_name,applications,approved,ROUND(rate,2) AS approval_rate_pct,
 DENSE_RANK() OVER(ORDER BY rate DESC) AS approval_rank FROM s ORDER BY approval_rank,branch_id;
