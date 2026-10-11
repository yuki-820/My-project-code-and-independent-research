-- 快照内分期金额偿付比
-- 不是平均每期偿付率，也不是含时间截断的某日催收回款率。
SELECT ROUND(SUM(amount_due),2) AS scheduled_amount,
 ROUND(SUM(amount_paid),2) AS paid_amount,
 ROUND(100.0*SUM(amount_paid)/NULLIF(SUM(amount_due),0),2) AS paid_ratio_pct
FROM repayments;
