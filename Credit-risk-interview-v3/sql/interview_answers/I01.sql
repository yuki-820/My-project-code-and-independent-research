-- Per-video valid-play completion rate. Deduplicate exact repeated event reports.
WITH clean AS (
 SELECT DISTINCT event_id,video_id,watch_sec FROM plays
 WHERE watch_sec IS NOT NULL AND watch_sec>=0
)
SELECT v.video_id,COUNT(c.event_id) AS valid_plays,
 SUM(CASE WHEN c.watch_sec>=v.duration_sec THEN 1 ELSE 0 END) AS complete_plays,
 ROUND(100.0*SUM(CASE WHEN c.watch_sec>=v.duration_sec THEN 1 ELSE 0 END)
       /NULLIF(COUNT(c.event_id),0),2) AS completion_rate_pct
FROM videos v LEFT JOIN clean c ON v.video_id=c.video_id
WHERE v.duration_sec>0
GROUP BY v.video_id ORDER BY v.video_id;
