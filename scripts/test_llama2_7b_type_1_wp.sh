python train.py \
    --mode test \
    --use_trans_e True \
    --bf16 True \
    --state_embed_size 4096 \
    --hidden_dim 4096 \
    --max_step_length 2 \
    --batch_size 4096 \
    --test_batch_size 128 \
    --stater_type llama \
    --instruction_type 1 \
    --stater_path /root/autodl-tmp/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc \
    --stater_cache_dir "" \
    --actor_checkpoint_path "runs/LLM-ARK/TRAINASSISTANT/2024-01-31_21-56-44/actor.pkt" \
    --rl_test_data_path datasets/OpenDialKG/Reason/test_type_1.json 
    