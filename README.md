# Data-driven approach of Urban vs. Rural Egyptian fruit bats differences

Prequisites:
- Download the data from https://data.mendeley.com/datasets/rn268h9cmy/1 is required prior running the scripts.
- requirements.txt contains the python environment used during development

There are two files in this project:

1. **model_train.py** for training
```usage: model_train.py [-h] -data DATA_PATH -output OUTPUT_DIR [-k K_FOLD] [-evaluate EVALUATE_MODEL]

Rural vs urban bats model training

options:
  -h, --help            show this help message and exit
  -data DATA_PATH, --data-path DATA_PATH
                        Path to sorted_data directory in dataset
  -output OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Path to output location
  -k K_FOLD, --k-fold K_FOLD
                        K for K-fold
  -evaluate EVALUATE_MODEL, --evaluate-model EVALUATE_MODEL
                        Path to (cross-validation) models for evaluation only
```

2. **visualize.py** for visualizing the results

```
usage: visualize.py [-h] -data DATA_PATH -k K_FOLD -model MODEL_PATH -vis VISUALIZATION_OUTPUT_PATH

Rural vs urban bats model Grad-CAM visualization

options:
  -h, --help            show this help message and exit
  -data DATA_PATH, --data-path DATA_PATH
                        Path to sorted_data directory in dataset
  -k K_FOLD, --k-fold K_FOLD
                        K for K-fold from training parameters
  -model MODEL_PATH, --model-path MODEL_PATH
                        Path to directory that contains the model
  -vis VISUALIZATION_OUTPUT_PATH, --visualization-output-path VISUALIZATION_OUTPUT_PATH
                        Path to output visualization files
```

Both files required arguments, including the path to the downloaded dataset.

Pretrained models are in "**test_trainer_2022-09-03_23-07-23_9**" directory. \
Visualization of the pretrained is in the **images** directory. \
Accompanied to that is the "**test_df_bats.html**" file. Open the html file to interactively view the visualization.
