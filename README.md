# Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study

## Abstract
Large language models have attracted significant attention due to their emergent abilities. However, their practical applications and deployments face notable challenges owing to their enormous scale. To overcome these challenges, several model compression methods have been proposed. Yet, it remains unclear whether the emergent abilities harnessed through model scale will diminish after compression.
This study aims to assess the existence of in-context learning, chain-of-thought reasoning, and instruction-following abilities in quantized large language models. 
Our empirical findings indicate that the abilities of the models remain largely unaffected by 4-bit quantization while 2-bit models encounter severe performance degradation. To improve the performance of low-bit models, we investigate two potential approaches: fine-grained quantization, which involves analyzing the impacts of model structures, and performance compensation through model fine-tuning.
Consequently, this research sheds light on the possibilities of employing extremely low-bit representations in expensive language models.

![Fine-tuning after quantization with GPTQ](/home/pyliu/projects/git_pro/EmpiricalStudy/figures/main.png)
## Setup

Install dependencies
```bash
pip install -r requirements.txt
cd peft/
pip install -e .
```

## Fine-tuning before quantization
For LoRA, change the `lora_r` by setting `lora_r={8,16,64}` for more trainable parameters. 
### LoRA
```bash
finetune_multi alpaca_lora $data_dir/alpaca_data_cleaned.json --adapter_name=lora\ --base_model=decapoda-research/llama-7b-hf\ --learning_rate=7e-5\ --use_gradient_checkpointing
```

## Fine-tuning after quantization
With the help of [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa), we can easily quantize 65B LLaMA into 2/4/8 bit precision.

### GPTQ-LoRA
- step 1: quantize the LLMs into low-bit precision. More details can refer to [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa).
```bash
CUDA_VISIBLE_DEVICES=1 python $base_dir/llama.py /mnt/data/pyliu/llama-65b-hf c4 --wbits 2 --true-sequential --act-order --groupsize 128 --save $out_dir/llama65b-2bit.pt $4
```
- step 2: post process the weights save to file:"$checkpoint/llama-65b-2bit-formulate" 
```python
state_dict = torch.load(quant_checkpoint, map_location="cpu")
new_state_dict = {}
for k,v in state_dict.items():
    new_state_dict["base_model.model." + k] = v
torch.save(new_state_dict, "llama-65b-4bit-formulate")
```
- step 2: bash run.sh
```bash
finetune_multi alpaca_gptqlora_65B_2bit $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"\ --base_model=$checkpoint/llama-65b-hf\ --quant_checkpoint="$checkpoint/llama-65b-4bit-formulate"\ --use_gradient_checkpointing\ --bits=2
```
# Resource Consumption
We present a table of resouce needed for fine-tuning on quantized model weights, which contains Trainable Parameters, GPU RAM Usage, and Fine-tuning Time on the alpaca dataset.

Hyperparameters: 
```bash
--batch_size 16
--micro_batch_size 2
--num_epochs 3
--learning_rate 3e-4
--target_modules "['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"
```

Hardware: 2*A100 GPUs

| Model                 | Bits   |Trainable Parameters  | GPU RAM Usage | Fine-tuning Time |
|-----------------------|--------|----------------------|---------------|------------------|
| LLaMA-7B-GPTQLoRA     | 2      | 20.0M                | 2.2GB         |     1h           | 
| LLaMA-7B-GPTQLoRA     | 4      | 20.0M                | 3.7GB         |     1h           | 
| LLaMA-65B-GPTQLoRA    | 2      | 21.0M                | 17.5GB        |     1h           | 
| LLaMA-65B-GPTQLoRA    | 4      | 21.0M                | 32.4GB        |     1h           | 

# Acknowledgements
Thankes to the powerful PTQ project [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa).

Thanks Meta AI for releasing LLaMA models.