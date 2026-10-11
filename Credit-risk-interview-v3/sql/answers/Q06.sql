-- 产品类型风险分布
-- 笔数率与金额率不同；没有剩余本金与回收台账，不能推算真实损失。
SELECT loan_type,COUNT(*) AS loans,SUM(loan_amount) AS original_principal,
 SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 ROUND(100.0*SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END)/COUNT(*),2) AS default_rate_pct
FROM loans GROUP BY loan_type ORDER BY original_principal DESC,loan_type;
