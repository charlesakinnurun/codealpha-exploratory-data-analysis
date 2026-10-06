# Data layout

* `data/raw/` — immutable source files (local only, git-ignored because ~1 GB total exceeds GitHub limits):
  * `events.csv` — 2,756,101 rows; `timestamp, visitorid, event, itemid, transactionid`
  * `category_tree.csv` — 1,669 rows; `categoryid, parentid` (25 roots)
  * `item_properties_part1.csv` — 10,999,999 rows (EAV: `timestamp, itemid, property, value`)
  * `item_properties_part2.csv` — 9,275,903 rows (same EAV schema)
* `data/processed/` — regenerable derivatives (git-ignored; rebuild with the stage scripts pattern):
  * `events_session30.parquet` — events + deterministic 30-min `session_id`
  * `prop_available.parquet` — 1,503,639 `available` timeline rows (417,053 items)
  * `prop_categoryid.parquet` — 788,214 `categoryid` timeline rows (417,053 items)
* No data dictionary or documentation was found in the repo. Only `available` and `categoryid` are face-interpretable; all other property codes are opaque.
* Timestamps are integer epoch **milliseconds** (events: May–Sep 2015). Convert with `src.data_processing.to_utc_datetime`.
* Working hypothesis (unconfirmed): RetailRocket-style e-commerce clickstream. Do not cite as fact.
