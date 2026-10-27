from result_plotter import plot_barchart
import pandas as pd

lstm_results = pd.read_csv("lstm_search_results.csv")
xgboost_results = pd.read_csv("xgboost_search_results.csv")
rf_results = pd.read_csv("rf_search_results.csv")
gru_results = pd.read_csv("gru_search_results.csv")

def latest_search_energy(df, method):
    matches = df.loc[
        df["search_method"].astype(str).str.lower() == method.lower(),
        "search_energy_kwh",
    ]

    return matches.iloc[-1]


result_metrics = {"xgb" : {
                            "Grid kWh" : latest_search_energy(xgboost_results, "grid"), 
                            "Random kWh" : latest_search_energy(xgboost_results, "random"), 
                            }, 
                "lstm" : {
                            "Grid kWh" : latest_search_energy(lstm_results, "grid"), 
                            "Random kWh" : latest_search_energy(lstm_results, "random"), 
                        },
                "rf" : {
                            "Grid kWh" : latest_search_energy(rf_results, "grid"), 
                            "Random kWh" : latest_search_energy(rf_results, "random"), 
                        },
                "gru" : {
                            "Grid kWh" : latest_search_energy(gru_results, "grid"), 
                            "Random kWh" : latest_search_energy(gru_results, "random"), 
                        }
                }


metrics = [ 
           "Grid kWh", 
           "Random kWh", 
           ]
    
plot_barchart(result_metrics, metrics)

