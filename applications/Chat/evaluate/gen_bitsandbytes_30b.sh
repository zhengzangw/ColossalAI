export CUDA_VISIBLE_DEVICES="6"
device_number=1
model_name="opt-bitsandbytes"
model_path="/home/lczxl/.cache/huggingface/hub/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546"
dataset="/home/lcxyc/data1/evaluate/ColossalAI/applications/Chat/evaluate/table/evaluation_dataset/cn/evaluate_cn_v4_300.json"
answer_path="opt-bitsandbytes-30b"


torchrun --standalone --nproc_per_node=$device_number generate_answers.py \
    --model 'a' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 2 \
    --answer_path $answer_path \
    --max_length 1024
