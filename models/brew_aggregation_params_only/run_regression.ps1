# Stop execution on any error
$ErrorActionPreference = "Stop"

# Data parameters
$PATH_INPUT = "data/processed/cof_selected_params_only_reg_brew.csv"
$PATH_OUTPUTS_DIR = "models/brew_aggregation_params_only/output"
$COLUMN_TARGET = "liking"

# Model selection parameters
$CLS_METHODS = "none"
$REG_METHODS = "svr,rf,ada,mlp"
$OD_METHODS = "none,iforest,lof"
$MVI_METHODS = "simple" # No missing values in this dataset
$FS_METHODS = "minmax,standard,none"
$OS_METHODS = "none" 
$N_FOLDS = 0
$SCORING = "r2"
# Reproducibility parameters
$RANDOM_STATE = 42

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
    --random-state $RANDOM_STATE `
    --mode "regression" `
    --nested
