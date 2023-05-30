export CUDA_VISIBLE_DEVICES="7"
device_number=1
model_name="opt-deepspeed-fp16"


# 6.7b
model_path="/home/lczxl/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0"
dataset="/home/lcxyc/data1/evaluate/ColossalAI/applications/Chat/evaluate/table/evaluation_dataset/cn/evaluate_cn_v4_300.json"
answer_path="opt-deepspeed-fp16-6.7b"
torchrun --standalone --nproc_per_node=$device_number deepspeed_generate.py \
    --model 'a' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 8 \
    --answer_path $answer_path \
    --max_length 1024


# 13b
model_path="/home/lczxl/.cache/huggingface/hub/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5"
dataset="/home/lcxyc/data1/evaluate/ColossalAI/applications/Chat/evaluate/table/evaluation_dataset/cn/evaluate_cn_v4_300.json"
answer_path="opt-deepspeed-fp16-13b"
torchrun --standalone --nproc_per_node=$device_number deepspeed_generate.py \
    --model 'a' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 4 \
    --answer_path $answer_path \
    --max_length 1024

# 30b
model_path="/home/lczxl/.cache/huggingface/hub/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546"
dataset="/home/lcxyc/data1/evaluate/ColossalAI/applications/Chat/evaluate/table/evaluation_dataset/cn/evaluate_cn_v4_300.json"
answer_path="opt-deepspeed-fp16-30b"
torchrun --standalone --nproc_per_node=$device_number deepspeed_generate.py \
    --model 'a' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 2 \
    --answer_path $answer_path \
    --max_length 1024
