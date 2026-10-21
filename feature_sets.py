FEATURE_SETS = {
    "full_set": [
    "hour",
    "price_value",
    "buy_volume_value",
    "sell_volume_value",
    "net_volume",

    "flow_DK1_export",
    "flow_DK1_import",
    "flow_GER_export",
    "flow_GER_import",
    "flow_NL_export",
    "flow_NL_import",
    "flow_NO1_export",
    "flow_NO1_import",
    "flow_NO5_export",
    "flow_NO5_import",

    "capacity_GER",
    "capacity_NL",
    "capacity_NO1",
    "capacity_NO5",

    "util_GER",
    "util_NL",
    "util_NO1",
    "util_NO5",

    "price_DK1",
    "price_NO1",
    "price_NO5",

    "price_NL",
    "price_DE_LU",
    "price_BE",
    "price_FR",

    "total_exports",
    "total_imports",
    "net_flow",

    "day_of_week",
    "month",
    "day_of_year",

    "day_sin",
    "day_cos",

    "year_sin",
    "year_cos",

    "price_lag_24",
    "price_lag_168",

    "price_DK1_lag24",
    "price_NO1_lag24",
    "price_NO5_lag24",

    "price_DK1_lag168",
    "price_NO1_lag168",
    "price_NO5_lag168",
],

    "lstm_set": [
    "hour",
    "price_value",
    "buy_volume_value",
    "sell_volume_value",
    "net_volume",

    "flow_DK1_export",
    "flow_DK1_import",
    "flow_GER_export",
    "flow_GER_import",
    "flow_NL_export",
    "flow_NL_import",
    "flow_NO1_export",
    "flow_NO1_import",
    "flow_NO5_export",
    "flow_NO5_import",

    "price_DK1",
    "price_NO1",
    "price_NO5",

    "price_NL",
    "price_DE_LU",
    "price_BE",
    "price_FR",

    "total_exports",
    "total_imports",
    "net_flow",

    "day_of_week",
    "month",
    "day_of_year",

    "day_sin",
    "day_cos",

    "year_sin",
    "year_cos",

    "price_lag_24",
    "price_lag_168",

    "price_DK1_lag24",
    "price_NO1_lag24",
    "price_NO5_lag24",

    "price_DK1_lag168",
    "price_NO1_lag168",
    "price_NO5_lag168",

    "Solar",
    "Wind Onshore",
    "Wind Offshore",

    "Forecasted Generation",
    "Forecasted Load",
],

    "minimal_set": [
        "price_value",
        "price_lag_24",
        "price_lag_168",
        "day_sin",
        "day_cos",
    ],

    "minimal_no_price": [
        "price_lag_24",
        "price_lag_168",
        "day_sin",
        "day_cos",
    ],

    "market_prices": [
        "price_value",
        "price_lag_24",
        "price_lag_168",
        "price_DK1",
        "price_NO1",
        "price_NO5",
        "price_NL",
        "price_DE_LU",
        "price_BE",
        "price_FR",
        "day_sin",
        "day_cos",
    ],

    # "reservoir": [
    #     "price_value",
    #     "price_lag_24",
    #     "price_lag_168",
    #     "no2_reservoir_level_MWh",
    #     "no1_reservoir_level_MWh",
    #     "no5_reservoir_level_MWh",
    #     "day_sin",
    #     "day_cos",
    # ],

    "renewables": [
        "price_value",
        "price_lag_24",
        "price_lag_168",
        "Solar",
        "Wind Onshore",
        "Wind Offshore",
        "Forecasted Generation",
        "Forecasted Load",
        "day_sin",
        "day_cos",
    ],
}