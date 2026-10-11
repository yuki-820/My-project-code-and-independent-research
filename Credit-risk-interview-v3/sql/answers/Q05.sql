-- 累计放款额
-- 先月度聚合再开窗；解释ROWS与RANGE。
WITH m AS (SELECT SUBSTR(disbursement_date,1,7) AS disb_month,SUM(loan_amount) AS amount
 FROM loans GROUP BY disb_month)
SELECT disb_month,amount,SUM(amount) OVER(ORDER BY disb_month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_amount
FROM m ORDER BY disb_month;
