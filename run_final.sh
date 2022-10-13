# STEP 1: Training
# To run an experiment, first set the parameters in `experiments.py` and take the exp id here
# The working directory is set to `./work_dirs_test`. Change it in run_experiments.py if you would like to.
## source only
python run_experiments.py --exp 0
## SePiCo, 3 random runs
python run_experiments.py --exp 1

# STEP 2: Evaluate and do palette
# This step evaluates the model and outputs corresponding test set labels (zero v2 val/test set)
# Gets visuals here
## source only
python -m tools.test CFG_PATH MODEL_PATH --show-dir ./PATH_TO_SHOW/exp0/palette --opacity 1
## ensemble sepico
python -m tools.ensemble_test CFG1_PATH CFG2_PATH CFG3_PATH --checkpoint MODEL1_PATH MODEL2_PATH MODEL3_PATH --show-dir ./PATH_TO_SHOW/exp1/palette --opacity 1 --ensemble-policy average_policy

# STEP 3: Converting for submission
# This step converts visual labels to train id labels to prepare for submission
python -m tools.convert_visuals_to_labels ./PATH_TO_SHOW/exp0/palette ./PATH_TO_SHOW/exp0/source_only
python -m tools.convert_visuals_to_labels ./PATH_TO_SHOW/exp1/palette ./PATH_TO_SHOW/exp1/uda


# STEP 4: Packing and zipping
zip -q -r sepico_v1_to_v2.zip ./PATH_TO_SHOW/exp1/uda ./PATH_TO_SHOW/exp0/source_only