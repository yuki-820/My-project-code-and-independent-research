-- JOIN本金重复放大
-- SUM(DISTINCT loan_amount)不是修复方案：不同贷款可能金额相同。先按loan_id聚合。
WITH x AS (SELECT COUNT(*) AS rows_after_join,SUM(l.loan_amount) AS inflated_amount
 FROM loans l JOIN repayments r USING(loan_id)), y AS (SELECT COUNT(*) AS loans,SUM(loan_amount) AS correct_amount FROM loans)
SELECT x.rows_after_join,y.loans,x.inflated_amount,y.correct_amount,
 ROUND(1.0*x.inflated_amount/y.correct_amount,2) AS inflation_multiple FROM x CROSS JOIN y;
