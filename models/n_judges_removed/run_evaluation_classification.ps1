# Stop execution on any error
$ErrorActionPreference = "Stop"

# Data parameters
$COLUMN_TARGET = "binary_liking"
$OVERRIDE_MODEL_INDEX = 0
$BOOTSTRAP = 0
$RANDOM_STATE = 42
$MODE = "binary" 

Get-ChildItem -Path "data/processed/n_judges_removed" -Filter "cof_selected_holdout_clf_noagg_*.csv" | 
ForEach-Object {
    $_.Name -match ".*?n(\d+)_(\d+)"
    $n = $Matches[1]
    $x = $Matches[2]
    $PATH_DATA_TEST = $_.FullName
    $PATH_DATA_BOOTSTRAP = "data/processed/n_judges_removed/cof_selected_all_clf_noagg_n$($n)_$($x).csv"
    $PATH_OUTPUTS_DIR = "models/n_judges_removed/output_clf_n$($n)_$($x)"

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

}