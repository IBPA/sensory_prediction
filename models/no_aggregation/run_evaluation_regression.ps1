# Stop execution on any error
$ErrorActionPreference = "Stop"

# Data parameters
$PATH_DATA_TEST = "data/processed/cof_selected_holdout_reg_noagg.csv"
$PATH_DATA_BOOTSTRAP = "data/processed/cof_selected_all_reg_noagg.csv"
$PATH_OUTPUTS_DIR = "models/no_aggregation/output_reg"
$COLUMN_TARGET = "liking"
$OVERRIDE_MODEL_INDEX = 0
$BOOTSTRAP = 10
$RANDOM_STATE = 42
$MODE = "regression"  

# Run preprocessing
python -m msap.run_evaluation `
    $PATH_DATA_TEST `
    $PATH_DATA_BOOTSTRAP `
    $PATH_OUTPUTS_DIR `
    --column-target $COLUMN_TARGET `
    --override-model-index $OVERRIDE_MODEL_INDEX `
    --bootstrap $BOOTSTRAP `
    --random-state $RANDOM_STATE `
    --mode $MODE 
