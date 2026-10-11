-- 组合规模与状态违约占比
-- 一行一笔贷款。原始本金之和不是当前贷款余额；违约依赖源status定义。
SELECT COUNT(*) AS total_loans, SUM(loan_amount) AS original_principal,
 SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) AS active_loans,
 SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END) AS defaulted_loans,
 ROUND(100.0*SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS default_rate_pct
FROM loans;
