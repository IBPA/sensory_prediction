# Stop execution on any error
$ErrorActionPreference = "Stop"

# Data parameters
$PATH_DATA_TEST = "data/processed/cof_selected_reg_brew.csv"
$PATH_DATA_BOOTSTRAP = "data/processed/cof_selected_reg_brew.csv"
$PATH_OUTPUTS_DIR = "models/brew_aggregation/output"
$COLUMN_TARGET = "liking"
$OVERRIDE_MODEL_INDEX = 0
$BOOTSTRAP = 100
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
