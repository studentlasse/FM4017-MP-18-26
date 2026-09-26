WITH Inflow AS (
	SELECT 
	to_zone_id,
	delivery_day,
	hour,
	SUM(CASE WHEN Flows.from_zone_id = 12 THEN Flows.flow_value ELSE 0 END) AS inflow_from_12,
    SUM(CASE WHEN Flows.from_zone_id = 16 THEN Flows.flow_value ELSE 0 END) AS inflow_from_16,
    SUM(CASE WHEN Flows.from_zone_id = 9 THEN Flows.flow_value ELSE 0 END) AS inflow_from_9,
    SUM(CASE WHEN Flows.from_zone_id = 10 THEN Flows.flow_value ELSE 0 END) AS inflow_from_10,
    SUM(CASE WHEN Flows.from_zone_id = 8 THEN Flows.flow_value ELSE 0 END) AS inflow_from_8,
    SUM(CASE WHEN Flows.from_zone_id = 7 THEN Flows.flow_value ELSE 0 END) AS inflow_from_7
    FROM Flows
    WHERE Flows.to_zone_id = 13
    GROUP BY
    Flows.delivery_day,
    Flows."hour", 
    Flows.to_zone_id
), 
Outflow AS (
	SELECT 
		from_zone_id,
		delivery_day,
		hour,
		SUM(CASE WHEN Flows.to_zone_id = 12 THEN Flows.flow_value ELSE 0 END) AS outflow_to_12,
    	SUM(CASE WHEN Flows.to_zone_id = 16 THEN Flows.flow_value ELSE 0 END) AS outflow_to_16,
    	SUM(CASE WHEN Flows.to_zone_id = 9 THEN Flows.flow_value ELSE 0 END) AS outflow_to_9,
    	SUM(CASE WHEN Flows.to_zone_id = 10 THEN Flows.flow_value ELSE 0 END) AS outflow_to_10,
    	SUM(CASE WHEN Flows.to_zone_id = 8 THEN Flows.flow_value ELSE 0 END) AS outflow_to_8,
    	SUM(CASE WHEN Flows.to_zone_id = 7 THEN Flows.flow_value ELSE 0 END) AS outflow_to_7
	FROM Flows 
	WHERE from_zone_id = 13
	GROUP BY
    Flows.delivery_day,
    Flows."hour", 
    Flows.from_zone_id
), 
Incapacity AS (
	SELECT 
	to_zone_id,
	delivery_day,
	hour,
	SUM(CASE WHEN Capacities.from_zone_id = 12 THEN Capacities.capacity_value ELSE 0 END) AS capacity_from_12,
    SUM(CASE WHEN Capacities.from_zone_id = 16 THEN Capacities.capacity_value ELSE 0 END) AS capacity_from_16,
    SUM(CASE WHEN Capacities.from_zone_id = 9 THEN Capacities.capacity_value ELSE 0 END) AS capacity_from_9,
    SUM(CASE WHEN Capacities.from_zone_id = 10 THEN Capacities.capacity_value ELSE 0 END) AS capacity_from_10,
    SUM(CASE WHEN Capacities.from_zone_id = 8 THEN Capacities.capacity_value ELSE 0 END) AS capacity_from_8,
    SUM(CASE WHEN Capacities.from_zone_id = 7 THEN Capacities.capacity_value ELSE 0 END) AS capacity_from_7
    FROM Capacities
    WHERE Capacities.to_zone_id = 13
    GROUP BY
    Capacities.delivery_day,
    Capacities."hour", 
    Capacities.to_zone_id
),
Outcapacity AS (
	SELECT 
	from_zone_id,
	delivery_day,
	hour,
	SUM(CASE WHEN Capacities.to_zone_id = 12 THEN Capacities.capacity_value ELSE 0 END) AS capacity_to_12,
    SUM(CASE WHEN Capacities.to_zone_id = 16 THEN Capacities.capacity_value ELSE 0 END) AS capacity_to_16,
    SUM(CASE WHEN Capacities.to_zone_id = 9 THEN Capacities.capacity_value ELSE 0 END) AS capacity_to_9,
    SUM(CASE WHEN Capacities.to_zone_id = 10 THEN Capacities.capacity_value ELSE 0 END) AS capacity_to_10,
    SUM(CASE WHEN Capacities.to_zone_id = 8 THEN Capacities.capacity_value ELSE 0 END) AS capacity_to_8,
    SUM(CASE WHEN Capacities.to_zone_id = 7 THEN Capacities.capacity_value ELSE 0 END) AS capacity_to_7
    FROM Capacities
    WHERE Capacities.from_zone_id = 13
    GROUP BY
    Capacities.delivery_day,
    Capacities."hour", 
    Capacities.from_zone_id
),
Buyvolumes AS (
	SELECT 
	delivery_day,
	hour,
	SUM(CASE WHEN Volumes.zone_id = 13 THEN Volumes.buy_volume_value ELSE 0 END) AS buyvolume_in_13,
	SUM(CASE WHEN Volumes.zone_id = 12 THEN Volumes.buy_volume_value ELSE 0 END) AS buyvolume_in_12,
    SUM(CASE WHEN Volumes.zone_id = 16 THEN Volumes.buy_volume_value ELSE 0 END) AS buyvolume_in_16,
    SUM(CASE WHEN Volumes.zone_id = 9 THEN Volumes.buy_volume_value ELSE 0 END) AS buyvolume_in__9,
    SUM(CASE WHEN Volumes.zone_id = 10 THEN Volumes.buy_volume_value ELSE 0 END) AS buyvolume_in_10,
    SUM(CASE WHEN Volumes.zone_id = 8 THEN Volumes.buy_volume_value ELSE 0 END) AS buyvolume_in_8,
    SUM(CASE WHEN Volumes.zone_id = 7 THEN Volumes.buy_volume_value ELSE 0 END) AS buyvolume_in_7
    FROM Volumes
    GROUP BY
    Volumes.delivery_day,
    Volumes."hour"
),
Sellvolumes AS (
	SELECT 
	delivery_day,
	hour,
	SUM(CASE WHEN Volumes.zone_id = 13 THEN Volumes.sell_volume_value ELSE 0 END) AS sellvolume_in_13,
	SUM(CASE WHEN Volumes.zone_id = 12 THEN Volumes.sell_volume_value ELSE 0 END) AS sellvolume_in_12,
    SUM(CASE WHEN Volumes.zone_id = 16 THEN Volumes.sell_volume_value ELSE 0 END) AS sellvolume_in_16,
    SUM(CASE WHEN Volumes.zone_id = 9 THEN Volumes.sell_volume_value ELSE 0 END) AS sellvolume_in__9,
    SUM(CASE WHEN Volumes.zone_id = 10 THEN Volumes.sell_volume_value ELSE 0 END) AS sellvolume_in_10,
    SUM(CASE WHEN Volumes.zone_id = 8 THEN Volumes.sell_volume_value ELSE 0 END) AS sellvolume_in_8,
    SUM(CASE WHEN Volumes.zone_id = 7 THEN Volumes.sell_volume_value ELSE 0 END) AS sellvolume_in_7
    FROM Volumes
    GROUP BY
    Volumes.delivery_day,
    Volumes."hour"
)
SELECT * FROM Prices 
LEFT JOIN Inflow 
ON Prices.delivery_day = Inflow.delivery_day 
AND Prices."hour" = Inflow."hour" 
LEFT JOIN Outflow 
ON Prices.delivery_day = Outflow.delivery_day 
AND Prices."hour" = Outflow."hour" 
LEFT JOIN Incapacity 
ON Prices.delivery_day = Incapacity.delivery_day 
AND Prices."hour" = Incapacity."hour" 
LEFT JOIN Outcapacity 
ON Prices.delivery_day = Outcapacity.delivery_day 
AND Prices."hour" = Outcapacity."hour" 
LEFT JOIN Buyvolumes  
ON Prices.delivery_day = Buyvolumes.delivery_day 
AND Prices."hour" = Buyvolumes."hour" 
LEFT JOIN Sellvolumes  
ON Prices.delivery_day = Sellvolumes.delivery_day 
AND Prices."hour" = Sellvolumes."hour" 
WHERE Prices.delivery_day >= '2015-01-01'
AND Prices.zone_id = 13