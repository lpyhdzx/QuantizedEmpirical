# training
base_dir=/home/pyliu/projects/git_pro/EmpiricalStudy
export OMP_NUM_THREADS=20
set -x
function finetune(){
    CUDA_VISIBLE_DEVICES=4 nohup python finetune.py \
        --base_model '/mnt/liupeiyu/llama_checkpoint/llama-7b/llama-7b' \
        --data_path $2 \
        --output_dir /mnt/liupeiyu/checkpoint/llm_adapters/$1 \
        --batch_size 16 \
        --micro_batch_size 4 \
        --num_epochs 3 \
        --learning_rate 3e-4 \
        --cutoff_len 256 \
        --val_set_size 120 \
        --adapter_name lora $3 > logs/$1_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
function finetune_multi(){
    WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=3225 finetune.py \
        --base_model '/mnt/data/pyliu/llama-7b-hf' \
        --data_path $2 \
        --output_dir /mnt/data/pyliu/checkpoint_pyliu/$1 \
        --batch_size 16 \
        --micro_batch_size 2 \
        --num_epochs 3 \
        --learning_rate 3e-4 \
        --cutoff_len 256 \
        --val_set_size 120 \
        --adapter_name lora $3 > logs/$1_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}

# test memory
# finetune_multi alpaca_gptqlora_65B_2bit_2gpu $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','v_proj']"\ --base_model=/mnt/data/pyliu/llama-65b-hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama65b-2bit-formulate"\ --use_gradient_checkpointing\ --bits=2
# finetune_multi alpaca_lora_7B_2bit_2gpu $base_dir/alpaca_data_cleaned.json --adapter_name=lora\ --target_modules="['q_proj','v_proj']"\ --base_model=/mnt/data/pyliu/llama-7b-hf\ --use_gradient_checkpointing\ --bits=2
# finetune_multi alpaca_lora_30B_2gpu $base_dir/alpaca_data_cleaned.json --adapter_name=lora\ --target_modules="['q_proj','v_proj']"\ --base_model=/mnt/data/pyliu/llama-30b-hf\ --use_gradient_checkpointing
# finetune_multi alpaca_lora_7B_2bit_2gpu $base_dir/alpaca_data_cleaned.json --adapter_name=lora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"\ --base_model=/mnt/data/pyliu/llama-7b-hf\ --use_gradient_checkpointing\ --bits=2
# finetune_multi alpaca_lora_7B_2bit_2gpu $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"\ --base_model=/mnt/data/pyliu/llama-7b-hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama-7b-2bit-formulate"\ --use_gradient_checkpointing\ --bits=2

# finetune_multi cot_gptqlora_65B_4bit $base_dir/CoT_data.json --adapter_name=gptqlora\ --target_modules="['q_proj','v_proj']"\ --base_model=/mnt/data/pyliu/llama-65b-hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama-65b-4bit-formulate"\ --use_gradient_checkpointing\ --bits=4
# finetune_multi alpaca_gptqlora_65B_2bit_qvdown $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"\ --base_model=/mnt/data/pyliu/llama-65b-hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama65b-2bit-formulate"\ --use_gradient_checkpointing\ --bits=2

finetune_multi alpaca_gptqlora_65B_4bit_qvdown $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"\ --base_model=/mnt/data/pyliu/llama-65b-hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama-65b-4bit-formulate"\ --use_gradient_checkpointing\ --bits=4
