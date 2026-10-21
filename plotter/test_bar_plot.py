from result_plotter import plot_barchart
import pandas as pd

lstm_results = pd.read_csv("lstm_results.csv")
xgboost_results = pd.read_csv("xgboost_results.csv")


result_metrics = {"xgb" : {
                            "RMSE" : xgboost_results.iloc[-1]['test_rmse'], 
                            "R2": xgboost_results.iloc[-1]['test_r2'], 
                            "MAE": xgboost_results.iloc[-1]['test_mae'],
                            "Train_kWh" : xgboost_results.iloc[-1]['training_energy_kwh'],
                            "Val_kWh" : xgboost_results.iloc[-1]['validation_energy_kwh'],
                            "Test_kWh" : xgboost_results.iloc[-1]['test_energy_kwh'],
                            }, 
                "lstm" : {
                            "RMSE" : lstm_results.iloc[-1]['test_rmse'], 
                            "R2": lstm_results.iloc[-1]['test_r2'], 
                            "MAE": lstm_results.iloc[-1]['test_mae'],
                            "Train_kWh" : lstm_results.iloc[-1]['training_energy_kwh'],
                            "Val_kWh" : lstm_results.iloc[-1]['validation_energy_kwh'],
                            "Test_kWh" : lstm_results.iloc[-1]['test_energy_kwh'],
                        }
                }


metrics = [ 
           "RMSE", 
           "R2", 
           "MAE", 
           "Train_kWh", 
           "Val_kWh", 
           "Test_kWh"
           ]
    
plot_barchart(result_metrics, metrics)

