# Stop execution on any error
$ErrorActionPreference = "Stop"

$COLUMN_TARGET = "liking"

# Model selection parameters
$CLS_METHODS = "none"
$REG_METHODS = "svr,rf,ada,mlp"
$OD_METHODS = "none,iforest,lof"
$MVI_METHODS = "simple" # No missing values in this dataset
$FS_METHODS = "minmax,standard,none"
$OS_METHODS = "none" 
$N_FOLDS = 5
$SCORING = "r2"

# Reproducibility parameters
$RANDOM_STATE = 42

Get-ChildItem -Path "data/processed/n_judges_removed" -Filter "cof_selected_train_reg_noagg_*.csv" | 
ForEach-Object {
    $_.Name -match ".*?n(\d+)_(\d+)"
    $n = $Matches[1]
    $x = $Matches[2]
    $PATH_INPUT = $_.FullName
    $PATH_OUTPUTS_DIR = "models/n_judges_removed/output_reg_n$($n)_$($x)"

    # Run preprocessing
    python -m msap.run_preprocess `
        $PATH_INPUT `
        $PATH_OUTPUTS_DIR `
        --column-target $COLUMN_TARGET `
        --od-methods $OD_METHODS `
        --mvi-methods $MVI_METHODS `
        --fs-methods $FS_METHODS `
        --random-state $RANDOM_STATE `
        --mode "regression"

    # Run grid search
    python -m msap.run_grid_search `
        $PATH_OUTPUTS_DIR `
        --column-target $COLUMN_TARGET `
        --cls-methods $CLS_METHODS `
        --reg-methods $REG_METHODS `
        --od-methods $OD_METHODS `
        --mvi-methods $MVI_METHODS `
        --fs-methods $FS_METHODS `
        --os-methods $OS_METHODS `
        --grid-search-n-splits $N_FOLDS `
        --grid-search-scoring $SCORING `
        --random-state $RANDOM_STATE
}
