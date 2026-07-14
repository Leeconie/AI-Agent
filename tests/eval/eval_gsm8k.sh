#!/bin/bash

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo ${script_dir}

export OPENAI_API_KEY=""

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

lm_eval \
    --model openai-chat-completions \
    --model_args model="vllm/custom",base_url="https://unifyapi.zeabur.app/v1/chat/completions",num_concurrent=8 \
    --tasks eval_gsm8k \
    --batch_size 8 \
    --apply_chat_template false \
    --output_path /tmp/results.json \
    --include_path ${script_dir}