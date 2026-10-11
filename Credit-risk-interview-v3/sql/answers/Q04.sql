-- 每月放款量
-- SQLite ISO日期取年月用SUBSTR；MySQL可用DATE_FORMAT。
SELECT SUBSTR(disbursement_date,1,7) AS disb_month,COUNT(*) AS loans,
 SUM(loan_amount) AS original_principal FROM loans GROUP BY disb_month ORDER BY disb_month;
