export CUDA_VISIBLE_DEVICES="0"

model_path=outputs/trm_plus_align_frozen/checkpoint-4950

python run_generation.py \
--model trm_plus \
--model_path=${model_path} \
--output_path generated/trm_plus_align_frozen.jsonl