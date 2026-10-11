-- 多笔借款客户
-- 包含已结清历史贷款，所以是历史累计原始本金，不是当前集中度敞口。
SELECT customer_id,COUNT(*) AS loans,SUM(loan_amount) AS cumulative_original_principal
FROM loans GROUP BY customer_id HAVING COUNT(*)>1 ORDER BY cumulative_original_principal DESC,customer_id;
