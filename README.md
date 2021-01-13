# LSTM with walk forward validation 

## Training 

1.  Install python +3.6.5 and virtualenv:
    [See here how to](https://pypi.org/project/virtualenv/).

2.  Create a virtualenv:
    ```
    user@server:~$ virtualenv -p python3 env
    ```
    
3.  Run virtualenv:
    ```
    user@server:~$ source env/bin/activate
    ```

4.  Run script to train:
    ```
    user@server:~$ python3 main.py
    ```
    
## Results


1.  A log file like this will be generate by every model INPUT X and OUTPUT Y:
```
Starting... cell 150, epoch 1000, batch_size 1000, input 4 and output 8
Training 4 8
split_dataset: train (11656, 108) test (2920, 108)
restructure_into_daily_data: train (11656, 108) test (2920, 108)
restructure_into_daily_data: train (1457, 8, 108) test (365, 8, 108)
evaluate_forecasts: real (365, 8) predict (365, 8)
summarize_scores: lstm: [11.493] 10.3, 12.1, 12.4, 12.2, 10.8, 10.6, 11.7, 11.7
Test RMSE: 11.493
ABS MIN: 0.0001609325408935547 MAX: 97.44740216732025 MEAN: 4.761863858710004 STD: 10.460350112658151


2. A csv file (test prediction dataset) will be generate for every model (ex: plot_results_4_8.csv) and plot a graph (open a windows browser) for every model show the results.




