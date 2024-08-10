# Sequential Batch Testing
Code used in computational experiments from the paper A General Framework for Sequential Batch-Testing




## Experiment results
The dependencies are listed in ```requirements.txt```.
Install with 
```zsh
% pip install -r requirements.txt
```
To replicate the experiments found in the paper, run

```zsh
% python sequential_batch_testing/experiment_runner.py
```

Experiment runner sets the random seed to ```0``` at the beginning of the run.

```experiment_runner.py``` dumps json result files into folder ```./results```.
Run the notebook ```results_tabulator.ipynb``` to process the results and generate the tables found in the paper.

The ```result``` folder and the outputs of ```results_tabulator``` is erased to ensure anonymity during double-anonymous review, as the file-path may be printed in the notebook. 
However, the exact values found in the paper can be obtained by re-running ```experiment_runner``` and ```results_tabulator``` on Python 3.12.4.


## Factor revealing problem
The factor problem found in chapter 3 is in the folder ```./LP```. Run the file ```sequential_testing.py``` to generate the values. 

