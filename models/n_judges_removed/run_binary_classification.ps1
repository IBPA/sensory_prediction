# Stop execution on any error
$ErrorActionPreference = "Stop"

$COLUMN_TARGET = "binary_liking"

# Model selection parameters
$CLS_METHODS = "svc,rf,ada,nb,mlp"
$REG_METHODS = "none"
$OD_METHODS = "lof,iforest,none" 
$MVI_METHODS = "simple" # MVI unneccessary for this dataset
$FS_METHODS = "minmax,standard,none"
$OS_METHODS = "none,smote" 
$N_FOLDS = 5
$SCORING = "f1"

# Reproducibility parameters
$RANDOM_STATE = 42

Get-ChildItem -Path "data/processed/n_judges_removed" -Filter "cof_selected_train_clf_noagg_*.csv" | 
ForEach-Object {
    $_.Name -match ".*?n(\d+)_(\d+)"
    $n = $Matches[1]
    $x = $Matches[2]
    $PATH_INPUT = $_.FullName
    $PATH_OUTPUTS_DIR = "models/n_judges_removed/output_clf_n$($n)_$($x)"

    # Run preprocessing
    python -m msap.run_preprocess `
        $PATH_INPUT `
        $PATH_OUTPUTS_DIR `
        --column-target $COLUMN_TARGET `
        --od-methods $OD_METHODS `
        --mvi-methods $MVI_METHODS `
        --fs-methods $FS_METHODS `
        --random-state $RANDOM_STATE `
        --mode "binary"

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