-- script that lists all shows from hbtn_0d_tvshows_rate by their rating
SELECT tv_shows.title, SUM(show_ratings.rate) AS rating
FROM tv_shows
JOIN show_ratings
ON tv_shows.id = show_ratings.show_id
GROUP BY tv_shows.title
ORDER BY rating DESC;
