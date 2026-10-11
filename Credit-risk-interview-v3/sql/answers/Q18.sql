-- 每支行原始本金前三名
-- 要恰好3条用ROW_NUMBER，要保留金额并列用DENSE_RANK并可能超过3条。
WITH s AS (SELECT branch_id,loan_id,loan_amount,
 ROW_NUMBER() OVER(PARTITION BY branch_id ORDER BY loan_amount DESC,loan_id) AS rn FROM loans)
SELECT branch_id,loan_id,loan_amount,rn FROM s WHERE rn<=3 ORDER BY branch_id,rn;
