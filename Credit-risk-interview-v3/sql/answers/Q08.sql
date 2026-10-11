-- 信用分分层
-- 同一客户可能多笔贷款；分母是贷款。当前分数没有历史可用时间，不能直接当申请时特征回测。
WITH s AS (SELECT l.status,c.credit_score,
 CASE WHEN c.credit_score<600 THEN '<600' WHEN c.credit_score<700 THEN '600-699'
 WHEN c.credit_score<750 THEN '700-749' WHEN c.credit_score<800 THEN '750-799' ELSE '800+' END AS band
 FROM loans l JOIN customers c USING(customer_id))
SELECT band,COUNT(*) AS loans,SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 ROUND(100.0*SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END)/COUNT(*),2) AS default_rate_pct
FROM s GROUP BY band ORDER BY MIN(credit_score);
