-- 放款月份分组与LAG
-- 是不同放款cohort在此快照的状态比例，不是当月新增违约率；账龄未对齐，不能断言策略改善。LAG默认上一行，缺月要补日历。
WITH m AS (SELECT SUBSTR(disbursement_date,1,7) AS disb_month,COUNT(*) AS loans,
 SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 100.0*SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END)/COUNT(*) AS rate
 FROM loans GROUP BY disb_month)
SELECT disb_month,loans,defaults,ROUND(rate,2) AS snapshot_default_rate_pct,
 ROUND(rate-LAG(rate) OVER(ORDER BY disb_month),2) AS previous_cohort_delta_pp
FROM m ORDER BY disb_month;
