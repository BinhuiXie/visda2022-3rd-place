# STEP 1: Training

# To run an experiment, first set the parameters in `mj_experiments.py` and take the exp id here
# e.g., for exp 1
CUDA_VISIBLE_DEVICES=0 python run_mj_experiments.py --exp 1
# The working directory is set to `./work_dirs_mj`. Change it in run_mj_experiments.py if you would like to.
# The previous logs are saved to `mjlogs`. My preference for this step is:
## CUDA_VISIBLE_DEVICES=0 nohup python run_mj_experiments.py --exp 1 > mjlogs/exp1.log >&1 &

# STEP 2: Evaluate and do palette

# This step evaluates the model and outputs corresponding test set labels (zero v2 val set for now)
# Gets visuals here
CUDA_VISIBLE_DEVICES=0 python -m tools.test \
"/mnt/data/bit/xbh/_visda2022/visda2022-ours/work_dirs_mj/local-exp1/220906_2122_zerov12zerov2_dacs_a999_fdthings_daformer_sepaspp_mitb5_poly10warm_s0_77205/220906_2122_zerov12zerov2_dacs_a999_fdthings_daformer_sepaspp_mitb5_poly10warm_s0_77205.json" \
"/mnt/data/bit/xbh/_visda2022/visda2022-ours/work_dirs_mj/local-exp1/220906_2122_zerov12zerov2_dacs_a999_fdthings_daformer_sepaspp_mitb5_poly10warm_s0_77205/latest.pth" \
--show-dir ./preds_mj/exp1/palette --opacity 1

# STEP 3: Converting for submission

# This step converts visual labels to train id labels to prepare for submission
CUDA_VISIBLE_DEVICES=0 python -m tools.convert_visuals_to_labels ./preds_mj/exp1/palette ./preds_mj/exp1/uda

# STEP 4: Packing and zipping
# First copy the source-only directory to `./preds_mj/exp1/`
# The SegFormer source-only folder can be found as "/mnt/data/bit/xbh/_visda2022/visda2022-ours/preds_mj/v1_to_v2_src_segf/"
cp -r ./preds_mj/v1_to_v2_src_segf ./preds_mj/exp1/source_only
# Alternatively, soft link can be used or copied
#1
## ln -s /mnt/data/bit/xbh/_visda2022/visda2022-ours/preds_mj/v1_to_v2_src_segf/ ./preds_mj/exp1/source_only
#2
## cp -d ./preds_mj/exp2/source_only ./preds_mj/exp1/

# STEP 5: Zip and upload
cd ./preds_mj/exp1/
zip -q -r exp1_v1_to_v2.zip ./uda ./source_only

# STEP 6: Submit
# exp1_v1_to_v2.zip is all you need to submit