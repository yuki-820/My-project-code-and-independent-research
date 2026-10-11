-- 申请放款漏斗核对
-- 先将loans聚合到申请粒度，防止一次审批对应多笔放款时重复计数。
WITH funded AS (SELECT application_id,COUNT(*) AS loan_count FROM loans GROUP BY application_id)
SELECT a.approval_status,COUNT(*) AS applications,
 SUM(CASE WHEN f.loan_count>0 THEN 1 ELSE 0 END) AS funded_applications,
 SUM(COALESCE(f.loan_count,0)) AS loans
FROM loan_applications a LEFT JOIN funded f USING(application_id)
GROUP BY a.approval_status ORDER BY a.approval_status;
