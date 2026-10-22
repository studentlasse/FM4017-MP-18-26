from calendar import c
import sqlite3
import numpy as np
import pandas as pd
import time
from datetime import datetime

DB_PATH = "Dataset/Nordpool/thesis_database.db"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

start_total = time.time()

log("Connecting to database...")
conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    p.delivery_day,
    p.hour,

    p.price_value,

    v.buy_volume_value,
    v.sell_volume_value,
    (v.buy_volume_value - v.sell_volume_value) AS net_volume,

    bz_other.zone_code AS connected_zone,

    CASE
        WHEN f.from_zone_id = p.zone_id
        THEN 'export'
        ELSE 'import'
    END AS flow_direction,

    f.flow_value,

    c.capacity_value,

    CASE
        WHEN c.capacity_value IS NULL
             OR c.capacity_value = 0
        THEN NULL
        ELSE f.flow_value * 1.0 / c.capacity_value
    END AS flow_utilization

FROM (
    -- Collapse 15-min quarter rows to hourly (no-op for hourly data)
    SELECT zone_id, delivery_day, hour,
           AVG(price_value) AS price_value
    FROM Prices
    GROUP BY zone_id, delivery_day, hour
) p

JOIN BiddingZones bz
    ON bz.zone_id = p.zone_id

LEFT JOIN (
    SELECT zone_id, delivery_day, hour,
           AVG(buy_volume_value)  AS buy_volume_value,
           AVG(sell_volume_value) AS sell_volume_value
    FROM Volumes
    GROUP BY zone_id, delivery_day, hour
) v
    ON v.zone_id = p.zone_id
   AND v.delivery_day = p.delivery_day
   AND v.hour = p.hour

LEFT JOIN (
    SELECT from_zone_id, to_zone_id, delivery_day, hour,
           AVG(flow_value) AS flow_value
    FROM Flows
    GROUP BY from_zone_id, to_zone_id, delivery_day, hour
) f
    ON (
           f.from_zone_id = p.zone_id
        OR f.to_zone_id = p.zone_id
       )
   AND f.delivery_day = p.delivery_day
   AND f.hour = p.hour

LEFT JOIN (
    SELECT from_zone_id, to_zone_id, delivery_day, hour,
           MAX(capacity_value) AS capacity_value
    FROM Capacities
    GROUP BY from_zone_id, to_zone_id, delivery_day, hour
) c
    ON c.from_zone_id = f.from_zone_id
   AND c.to_zone_id = f.to_zone_id
   AND c.delivery_day = f.delivery_day
   AND c.hour = f.hour

LEFT JOIN BiddingZones bz_other
    ON bz_other.zone_id =
       CASE
           WHEN f.from_zone_id = p.zone_id
           THEN f.to_zone_id
           ELSE f.from_zone_id
       END

WHERE bz.zone_code = 'NO2'
  AND bz_other.zone_code IN ('DK1', 'GER', 'NL', 'NO1', 'NO5')

ORDER BY
    p.delivery_day,
    p.hour;
"""

neighbor_price_query = """
SELECT
    p.delivery_day,
    p.hour,
    bz.zone_code,
    p.price_value
FROM (
    SELECT zone_id, delivery_day, hour,
           AVG(price_value) AS price_value
    FROM Prices
    GROUP BY zone_id, delivery_day, hour
) p
JOIN BiddingZones bz
    ON bz.zone_id = p.zone_id
WHERE bz.zone_code IN ('NO1', 'NO5', 'DK1', 'GER', 'NL');
"""


log("Executing SQL query...")
query_start = time.time()

flows = pd.read_sql_query(query, conn)

neighbor_prices = pd.read_sql_query(
    neighbor_price_query,
    conn
)

log(f"SQL query completed in {time.time()-query_start:.2f}s")

log("Creating datetime column...")
step = time.time()

flows["delivery_day"] = pd.to_datetime(flows["delivery_day"])
neighbor_prices["delivery_day"] = pd.to_datetime(
    neighbor_prices["delivery_day"]
)

log(f"Completed in {time.time()-step:.2f}s")

log("Loading Entso-E forecast data...")
step = time.time()

ENTSOE_PATH = "Dataset/Entso-E/entsoe_data_hourly.csv"

# First column is an unnamed index holding the UTC timestamps
entsoe = pd.read_csv(ENTSOE_PATH, index_col=0)

entsoe.index = pd.to_datetime(entsoe.index, utc=True)

# Nordpool data is in local Oslo time (delivery_day + hour),
# so convert the UTC timestamps before deriving the merge keys
entsoe.index = entsoe.index.tz_convert("Europe/Oslo")

entsoe["delivery_day"] = entsoe.index.tz_localize(None).normalize()
entsoe["hour"] = entsoe.index.hour

# DST fall-back days have a duplicated local hour (2A/2B);
# average them so the merge stays 1:1, mirroring the DB's quarter collapse
entsoe_hourly = (
    entsoe
    .groupby(["delivery_day", "hour"], as_index=False)
    .mean(numeric_only=True)
)

log(f"Completed in {time.time()-step:.2f}s")

log("creating base...")
step = time.time()

base = (
    flows[
        [
            "delivery_day",
            "hour",
            "price_value",
            "buy_volume_value",
            "sell_volume_value",
            "net_volume",
        ]
    ]
    .drop_duplicates()
)

log("Building flow features...")
step = time.time()

flow_features = (
    flows.pivot_table(
        index=["delivery_day", "hour"],
        columns=["connected_zone", "flow_direction"],
        values="flow_value",
        aggfunc="first"
    )
    .fillna(0)
)


flow_features.columns = [
    f"flow_{zone}_{direction}"
    for zone, direction in flow_features.columns
]

log(f"Completed in {time.time()-step:.2f}s")

log("Building capacity features...")
step = time.time()

capacity_features = (
    flows.pivot_table(
        index=["delivery_day", "hour"],
        columns="connected_zone",
        values="capacity_value",
        aggfunc="first"
    )
)

capacity_features.columns = [
    f"capacity_{c}"
    for c in capacity_features.columns
]

log(f"Completed in {time.time()-step:.2f}s")

log("Building utilization features...")
step = time.time()

util_features = (
    flows.pivot_table(
        index=["delivery_day", "hour"],
        columns="connected_zone",
        values="flow_utilization",
        aggfunc="first"
    )
    .fillna(0)
)

util_features.columns = [
    f"util_{c}"
    for c in util_features.columns
]

log(f"Completed in {time.time()-step:.2f}s")

log("Building neighbor price features...")
step = time.time()

neighbor_price_features = (
    neighbor_prices.pivot_table(
        index=["delivery_day", "hour"],
        columns="zone_code",
        values="price_value",
        aggfunc="first"
    )
)

neighbor_price_features.columns = [
    f"price_{c}"
    for c in neighbor_price_features.columns
]

log(f"Completed in {time.time()-step:.2f}s")

log("Joining features into dataset...")
step = time.time()

dataset = (
    base
    .set_index(["delivery_day", "hour"])
    .join(flow_features)
    .join(capacity_features)
    .join(util_features)
    .join(neighbor_price_features)
    .reset_index()
)

# merged here, before the feature engineering below, so the Entso-E prices
# get lags like the Nordpool ones
dataset = dataset.merge(
    entsoe_hourly,
    on=["delivery_day", "hour"],
    how="left"
)

log(f"Completed in {time.time()-step:.2f}s")

log("Building aggregate and calendar features...")
step = time.time()

export_cols = [
    c for c in dataset.columns
    if c.endswith("_export")
]

import_cols = [
    c for c in dataset.columns
    if c.endswith("_import")
]

dataset["total_exports"] = dataset[export_cols].sum(axis=1)

dataset["total_imports"] = dataset[import_cols].sum(axis=1)

dataset["net_flow"] = (
    dataset["total_imports"]
    - dataset["total_exports"]
)

dataset["day_of_week"] = (
    dataset["delivery_day"].dt.dayofweek
)

dataset["month"] = (
    dataset["delivery_day"].dt.month
)

dataset["day_of_year"] = (
    dataset["delivery_day"].dt.dayofyear
)

dataset["day_sin"] = (
    np.sin(2 * np.pi * dataset["hour"] / 24)
)

dataset["day_cos"] = (
    np.cos(2 * np.pi * dataset["hour"] / 24)
)

dataset["week_sin"] = (
    np.sin(2 * np.pi * dataset["day_of_week"] / 7)
)

dataset["week_cos"] = (
    np.cos(2 * np.pi * dataset["day_of_week"] / 7)
)

dataset["year_sin"] = (
    np.sin(2 * np.pi * dataset["day_of_year"] / 365)
)

dataset["year_cos"] = (
    np.cos(2 * np.pi * dataset["day_of_year"] / 365)
)

log(f"Completed in {time.time()-step:.2f}s")

log("Building lag features...")
step = time.time()

dataset = dataset.sort_values(
    ["delivery_day", "hour"]
)

dataset["price_lag_24"] = (
    dataset["price_value"].shift(24)
)

dataset["price_lag_48"] = (
    dataset["price_value"].shift(48)
)

dataset["price_lag_72"] = (
    dataset["price_value"].shift(72)
)

dataset["price_lag_168"] = (
    dataset["price_value"].shift(168)
)

price_cols = [
    c for c in dataset.columns
    if (c.startswith("price_") and "_lag" not in c and "_value" not in c) or ("forecast" in c)
]
log(f"creating lag for columns: {price_cols}")

for col in price_cols:
    dataset[f"{col}_lag1"] = dataset[col].shift(1)

for col in price_cols:
    dataset[f"{col}_lag2"] = dataset[col].shift(2)

for col in price_cols:
    dataset[f"{col}_lag3"] = dataset[col].shift(3)

for col in price_cols:
    dataset[f"{col}_lag24"] = dataset[col].shift(24)

for col in price_cols:
    dataset[f"{col}_lag48"] = dataset[col].shift(48)

for col in price_cols:
    dataset[f"{col}_lag72"] = dataset[col].shift(72)

for col in price_cols:
    dataset[f"{col}_lag168"] = dataset[col].shift(168)

log(f"Completed in {time.time()-step:.2f}s")

step = time.time()

ENTSOE_START = pd.Timestamp("2023-10-01")
before = len(dataset)
dataset = dataset[dataset["delivery_day"] >= ENTSOE_START].copy()
log(f"Dropped {before - len(dataset):,} rows before {ENTSOE_START.date()}")

output_file = "ml_dataset.csv"
dataset.to_csv(output_file, index=False)

log(f"CSV saved in {time.time()-step:.2f}s")

log("FINAL STATS")
log(f"Final rows: {len(dataset):,}")
log(f"Final columns: {len(dataset.columns)}")
log(f"Total runtime: {time.time()-start_total:.2f}s")

conn.close()
log("Done.")