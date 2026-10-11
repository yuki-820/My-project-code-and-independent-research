-- Paid gross GMV, includes today and previous 6 calendar days. No refund table in fixture.
WITH RECURSIVE calendar(day) AS (
 SELECT '2026-10-01'
 UNION ALL SELECT DATE(day,'+1 day') FROM calendar WHERE day<'2026-10-10'
), daily AS (
 SELECT paid_date,SUM(amount) AS gmv FROM orders WHERE status='paid' GROUP BY paid_date
), filled AS (
 SELECT c.day,COALESCE(d.gmv,0) AS gmv FROM calendar c LEFT JOIN daily d ON c.day=d.paid_date
)
SELECT day,gmv,SUM(gmv) OVER(ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS gmv_7d
FROM filled ORDER BY day;
