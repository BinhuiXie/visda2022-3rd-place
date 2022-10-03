# 10.3
CUDA_VISIBLE_DEVICES=0 nohup python run_test_experiments.py --exp 0 >testlogs/exp0.log >&1 &
CUDA_VISIBLE_DEVICES=1 nohup python run_test_experiments.py --exp 1 >testlogs/exp1.log >&1 &
CUDA_VISIBLE_DEVICES=2 nohup python run_test_experiments.py --exp 2 >testlogs/exp2.log >&1 &
CUDA_VISIBLE_DEVICES=3 python run_test_experiments.py --exp 3 >testlogs/exp3.log >&1 &
CUDA_VISIBLE_DEVICES=4 python run_test_experiments.py --exp 4 >testlogs/exp4.log >&1 &
CUDA_VISIBLE_DEVICES=5 python run_test_experiments.py --exp 5 >testlogs/exp5.log >&1 &
CUDA_VISIBLE_DEVICES=6 python run_test_experiments.py --exp 6 >testlogs/exp6.log >&1 &
CUDA_VISIBLE_DEVICES=7 python run_test_experiments.py --exp 7 >testlogs/exp7.log >&1 &




# STEP 0: checkout & pull
# if you uploaded files via pyCharm, just run `checkout.sh`
sh checkout.sh

# STEP 1: Training

# To run an experiment, first set the parameters in `test_experiments.py` and take the exp id here
# The working directory is set to `./work_dirs_test`. Change it in run_test_experiments.py if you would like to.
# The previous logs are saved to `testlogs`. My preference for this step is:
CUDA_VISIBLE_DEVICES=0 nohup python run_test_experiments.py --exp 0 >testlogs/exp0.log >&1 &


# STEP 2: Evaluate and do palette

# This step evaluates the model and outputs corresponding test set labels (zero v2 val set for now)
# Gets visuals here
CUDA_VISIBLE_DEVICES=0 python -m tools.test configs/daformer/zerov1all_to_zerov2_daformer_mit5.py work_dirs_test/local-exp0/xxxxx/latest.pth --show-dir ./final_test/exp0/palette --opacity 1

# STEP 3: Converting for submission

# This step converts visual labels to train id labels to prepare for submission
CUDA_VISIBLE_DEVICES=0 python -m tools.convert_visuals_to_labels ./final_test/exp0/palette ./final_test/exp0/uda

# STEP 4: Packing and zipping
# First copy the source-only directory to `./final_test/exp0/`
# The SegFormer source-only folder can be found as "/mnt/data/bit/xbh/_visda2022/visda2022-ours/final_test/v1_to_v2_src_segf/"
cp -r ./final_test/v1_to_v2_src_segf ./final_test/exp0/source_only
# Alternatively, soft link can be used or copied
#1
## ln -s /mnt/data/bit/xbh/_visda2022/visda2022-ours/final_test/v1_to_v2_src_segf/ ./final_test/exp0/source_only
#2
## cp -d ./final_test/exp2/source_only ./final_test/exp0/
# Then the required folders can be zipped as follows
cd ./final_test/exp0/
zip -q -r exp0_v1_to_v2.zip ./uda ./source_only

# STEP 5: Submit
# exp0_v1_to_v2.zip is all you need to submit