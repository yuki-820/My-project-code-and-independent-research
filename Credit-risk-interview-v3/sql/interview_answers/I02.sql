-- A duplicate login on the same date must count only once.
WITH days AS (SELECT DISTINCT user_id,login_date FROM logins),
numbered AS (
 SELECT user_id,login_date,
 ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY login_date) AS rn FROM days
), islands AS (
 SELECT user_id,login_date,DATE(login_date,'-'||rn||' days') AS island FROM numbered
)
SELECT user_id,MIN(login_date) AS start_date,MAX(login_date) AS end_date,COUNT(*) AS consecutive_days
FROM islands GROUP BY user_id,island HAVING COUNT(*)>=3 ORDER BY user_id,start_date;
