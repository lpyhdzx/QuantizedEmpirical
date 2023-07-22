## Install HuggingFace Transformers from source
# pip install git+https://github.com/huggingface/transformers
# cd transformers

python convert_llama_weights_to_hf.py \
    --input_dir /mnt/data/pyliu/llama2 --model_size 70B --output_dir /mnt/data/pyliu/llama2/llama2_70B_hf