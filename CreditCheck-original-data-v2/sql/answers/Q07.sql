-- 复核原作者DPD桶
-- 复现missed中最大dpd。原文90+标签实际是>90；此处修正为91+。未提供统一快照日期，不称为独立重算当前DPD。
WITH w AS (SELECT loan_id,MAX(CASE WHEN payment_status='missed' THEN dpd ELSE 0 END) AS reported_dpd
 FROM repayments GROUP BY loan_id), b AS (
 SELECT l.loan_id,l.loan_amount,
 CASE WHEN COALESCE(w.reported_dpd,0)=0 THEN '0'
 WHEN w.reported_dpd<=30 THEN '1-30' WHEN w.reported_dpd<=60 THEN '31-60'
 WHEN w.reported_dpd<=90 THEN '61-90' ELSE '91+' END AS bucket,
 COALESCE(w.reported_dpd,0) AS dpd
 FROM loans l LEFT JOIN w USING(loan_id) WHERE l.status<>'closed')
SELECT bucket,COUNT(*) AS loans,SUM(loan_amount) AS original_principal_proxy
FROM b GROUP BY bucket ORDER BY MIN(dpd);
