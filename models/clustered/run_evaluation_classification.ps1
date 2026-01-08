# Stop execution on any error
$ErrorActionPreference = "Stop"

# Data parameters
$PATH_DATA_TEST = "data/processed/cof_selected_holdout_clf_noagg_c1.csv"
$PATH_DATA_BOOTSTRAP = "data/processed/cof_selected_all_clf_noagg_c1.csv"
$PATH_OUTPUTS_DIR = "models/clustered/output_c1_clf"
$COLUMN_TARGET = "binary_liking"
$OVERRIDE_MODEL_INDEX = 0
$BOOTSTRAP = 100
$RANDOM_STATE = 42
$MODE = "binary" 

echo "Cluster 1"
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


# Data parameters
$PATH_DATA_TEST = "data/processed/cof_selected_holdout_clf_noagg_c2.csv"
$PATH_DATA_BOOTSTRAP = "data/processed/cof_selected_all_clf_noagg_c2.csv"
$PATH_OUTPUTS_DIR = "models/clustered/output_c2_clf"
$OVERRIDE_MODEL_INDEX = 0

echo "Cluster 2"
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