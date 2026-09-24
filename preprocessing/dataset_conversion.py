import sqlite3
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
WITH flow_features AS (
    SELECT
        bz.zone_id,
        f.delivery_day,
        f.hour,
        SUM(
            CASE
                WHEN f.to_zone_id = bz.zone_id
                THEN f.flow_value
                ELSE 0
            END
        ) AS incoming_flow,
        SUM(
            CASE
                WHEN f.from_zone_id = bz.zone_id
                THEN f.flow_value
                ELSE 0
            END
        ) AS outgoing_flow
    FROM BiddingZones bz
    LEFT JOIN flows f
        ON bz.zone_id IN (f.from_zone_id, f.to_zone_id)
    GROUP BY
        bz.zone_id,
        f.delivery_day,
        f.hour
),

capacity_features AS (
    SELECT
        bz.zone_id,
        c.delivery_day,
        c.hour,
        SUM(
            CASE
                WHEN c.from_zone_id = bz.zone_id
                THEN c.capacity_value
                ELSE 0
            END
        ) AS export_capacity,
        SUM(
            CASE
                WHEN c.to_zone_id = bz.zone_id
                THEN c.capacity_value
                ELSE 0
            END
        ) AS import_capacity
    FROM BiddingZones bz
    LEFT JOIN capacities c
        ON bz.zone_id IN (c.from_zone_id, c.to_zone_id)
    GROUP BY
        bz.zone_id,
        c.delivery_day,
        c.hour
)

SELECT
    p.zone_id,
    bz.zone_code,
    p.delivery_day,
    p.hour,
    p.price_value,
    v.buy_volume_value,
    v.sell_volume_value,
    (v.buy_volume_value - v.sell_volume_value) AS net_volume,

    COALESCE(ff.incoming_flow, 0) AS incoming_flow,
    COALESCE(ff.outgoing_flow, 0) AS outgoing_flow,
    COALESCE(ff.incoming_flow, 0) -
    COALESCE(ff.outgoing_flow, 0) AS net_flow,

    COALESCE(cf.import_capacity, 0) AS import_capacity,
    COALESCE(cf.export_capacity, 0) AS export_capacity,

    CAST(strftime('%w', p.delivery_day) AS INTEGER) AS day_of_week,
    CAST(strftime('%m', p.delivery_day) AS INTEGER) AS month,
    CAST(strftime('%j', p.delivery_day) AS INTEGER) AS day_of_year

FROM prices p
JOIN BiddingZones bz
    ON bz.zone_id = p.zone_id

LEFT JOIN volumes v
    ON p.zone_id = v.zone_id
   AND p.delivery_day = v.delivery_day
   AND p.hour = v.hour

LEFT JOIN flow_features ff
    ON p.zone_id = ff.zone_id
   AND p.delivery_day = ff.delivery_day
   AND p.hour = ff.hour

LEFT JOIN capacity_features cf
    ON p.zone_id = cf.zone_id
   AND p.delivery_day = cf.delivery_day
   AND p.hour = cf.hour

WHERE bz.zone_code = 'NO2'

ORDER BY
    p.delivery_day,
    p.hour;
"""

log("Executing SQL query...")
query_start = time.time()

df = pd.read_sql_query(query, conn)

log(f"SQL query completed in {time.time()-query_start:.2f}s")
log(f"Rows loaded: {len(df):,}")
log(f"Columns loaded: {len(df.columns)}")

memory_mb = df.memory_usage(deep=True).sum() / 1024**2
log(f"Memory usage: {memory_mb:.2f} MB")

log("Creating datetime column...")
step = time.time()

df["datetime"] = (
    pd.to_datetime(df["delivery_day"])
    + pd.to_timedelta(df["hour"] - 1, unit="h")
)

log(f"Completed in {time.time()-step:.2f}s")

log("Sorting dataset...")
step = time.time()

df = df.sort_values(["zone_id", "datetime"])

log(f"Completed in {time.time()-step:.2f}s")

lags = [1, 2, 3, 24, 48, 168] #hours

for lag in lags:
    log(f"Creating price_lag_{lag}...")
    step = time.time()

    df[f"price_lag_{lag}"] = (
        df.groupby("zone_id")["price_value"]
        .shift(lag)
    )

    log(f"price_lag_{lag} completed in {time.time()-step:.2f}s")

log("Creating rolling mean 24h...")
step = time.time()

df["price_ma_24"] = (
    df.groupby("zone_id")["price_value"]
      .transform(lambda x: x.rolling(24).mean())
)

log(f"Completed in {time.time()-step:.2f}s")

log("Creating rolling mean 168h...")
step = time.time()

df["price_ma_168"] = (
    df.groupby("zone_id")["price_value"]
      .transform(lambda x: x.rolling(168).mean())
)

log(f"Completed in {time.time()-step:.2f}s")

log("Creating target variable...")
step = time.time()

df["target_price_next_hour"] = (
    df.groupby("zone_id")["price_value"]
      .shift(-1)
)

log(f"Completed in {time.time()-step:.2f}s")

rows_before = len(df)

log("Dropping rows with NaN values...")
step = time.time()

df = df.dropna()

rows_after = len(df)

log(f"Rows before dropna: {rows_before:,}")
log(f"Rows after dropna : {rows_after:,}")
log(f"Rows removed      : {rows_before - rows_after:,}")
log(f"Completed in {time.time()-step:.2f}s")

output_file = "ml_dataset.csv"

log(f"Saving dataset to {output_file}...")
step = time.time()

df.to_csv(output_file, index=False)

log(f"CSV saved in {time.time()-step:.2f}s")

memory_mb = df.memory_usage(deep=True).sum() / 1024**2

log("FINAL STATS")
log(f"Final rows: {len(df):,}")
log(f"Final columns: {len(df.columns)}")
log(f"Final memory usage: {memory_mb:.2f} MB")
log(f"Total runtime: {time.time()-start_total:.2f}s")

conn.close()

log("Done.")