# Stop execution on any error
$ErrorActionPreference = "Stop"

# Data parameters
$PATH_INPUT = "data/processed/cof_selected_params_only_clf_brew.csv"
$PATH_OUTPUTS_DIR = "models/brew_aggregation_params_only/output"
$COLUMN_TARGET = "binary_liking"

# Model selection parameters
$CLS_METHODS = "svc,rf,ada,nb,mlp"
$REG_METHODS = "none"
$OD_METHODS = "none" 
$MVI_METHODS = "simple" # MVI unneccessary for this dataset
$FS_METHODS = "minmax,standard,none"
$OS_METHODS = "none" 
$N_FOLDS = 0
$SCORING = "f1"

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
    --random-state $RANDOM_STATE `
    --mode "binary" `
    --nested
