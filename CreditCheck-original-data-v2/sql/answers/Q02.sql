-- 申请审批通过率
-- 分母用全部申请，不能从仅有放款的loans表算通过率。
SELECT COUNT(*) AS applications,
 SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END) AS approved,
 ROUND(100.0*SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS approval_rate_pct
FROM loan_applications;
