-- Historical months are real columns reshaped into rows, not generated observations.
-- Keep labels OUT of this feature query.
WITH h AS (
 SELECT client_id,
 MAX(CASE WHEN month='2005-09' THEN pay_status END) AS latest_status,
 MAX(CASE WHEN month='2005-09' THEN bill_amount END) AS latest_bill,
 MAX(CASE WHEN month='2005-04' THEN bill_amount END) AS earliest_bill,
 MAX(CASE WHEN pay_status>0 THEN pay_status ELSE 0 END) AS worst_delay6,
 SUM(CASE WHEN pay_status>0 THEN 1 ELSE 0 END) AS delay_months6,
 SUM(CASE WHEN month>='2005-07' AND pay_status>0 THEN 1 ELSE 0 END) AS delay_months3,
 AVG(payment_amount) AS mean_paid6,
 SUM(CASE WHEN month>='2005-07' THEN payment_amount ELSE -payment_amount END) AS recent_minus_old_paid
 FROM monthly_history GROUP BY client_id
)
SELECT c.client_id,c.credit_limit,h.latest_status,
 1.0*h.latest_bill/c.credit_limit AS latest_util,
 h.worst_delay6,h.delay_months6,h.delay_months3,
 1.0*(h.latest_bill-h.earliest_bill)/c.credit_limit AS bill_trend6,
 1.0*h.mean_paid6/c.credit_limit AS mean_paid_to_limit6,
 1.0*h.recent_minus_old_paid/c.credit_limit AS paid_trend6
FROM clients c JOIN h USING(client_id) ORDER BY c.client_id;
