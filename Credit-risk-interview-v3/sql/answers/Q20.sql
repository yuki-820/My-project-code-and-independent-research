-- 状态与DPD交叉核对
-- on_time允许dpd为1或2，可能存在宽限口径，但作者未说明。记录差异，不自动改为late。
SELECT payment_status,COUNT(*) AS installments,MIN(dpd) AS min_dpd,MAX(dpd) AS max_dpd,
 SUM(CASE WHEN dpd>0 THEN 1 ELSE 0 END) AS positive_dpd_rows,
 SUM(CASE WHEN paid_date IS NULL THEN 1 ELSE 0 END) AS missing_paid_date_rows
FROM repayments GROUP BY payment_status ORDER BY payment_status;
