-- 每个客户最新申请
-- 日期并列用application_id稳定排序；MAX日期再JOIN可能一人多行。
WITH s AS (SELECT application_id,customer_id,application_date,approval_status,
 ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY application_date DESC,application_id DESC) AS rn
 FROM loan_applications)
SELECT customer_id,application_id,application_date,approval_status FROM s WHERE rn=1 ORDER BY customer_id;
