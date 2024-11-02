-- script that lists all genres in the database  by their rating
SELECT tv_genres.name AS name, SUM(rate) AS rating
FROM tv_genres
JOIN tv_show_genres
    ON show_genres.genre_id = tv_genres.id
JOIN show_ratings
    ON show_ratings.show_id = show_genres.show_id
GROUP BY name
ORDER BY rating DESC;
