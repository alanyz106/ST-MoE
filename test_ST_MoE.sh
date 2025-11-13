python main_Mocap.py --epoch 100 --num_stage 3 --depth 1 --nlayer 1 --w_tp 1 --w_sp 1 \
--model ST_MoE \
--top_k 4 \
--dataset Mocap \
--test_batch 100 \
--mode test \
--cudaid 5
# --log_name original_loss \

# --isSavePredictionstoNpy 
