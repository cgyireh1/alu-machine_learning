-- script that displays avg temperature(Fahrenheit) by city ordered
SELECT city, AVG(value) AS avg_temp
FROM temperatures GROUP BY city
ORDER BY avg_temp DESC;
