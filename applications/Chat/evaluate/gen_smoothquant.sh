export CUDA_VISIBLE_DEVICES="7"
device_number=1
model_name="opt-smoothquant"
dataset="/home/lcxyc/data1/evaluate/ColossalAI/applications/Chat/evaluate/table/evaluation_dataset/cn/evaluate_cn_v4_300.json"


# 6.7b
# model_path="/home/lczxl/.cache/huggingface/hub/models--mit-han-lab--opt-6.7b-smoothquant/snapshots/21daf5a055136f5a86c8a8e03bb28ba9ecb3a953"
# answer_path="opt-smoothquant-6.7b"
# torchrun --standalone --nproc_per_node=$device_number generate_answers.py \
#     --model 'a' \
#     --strategy ddp \
#     --model_path $model_path \
#     --model_name $model_name \
#     --dataset $dataset \
#     --batch_size 8 \
#     --answer_path $answer_path \
#     --max_length 1024


# 13b
model_path="/home/lczxl/.cache/huggingface/hub/models--mit-han-lab--opt-13b-smoothquant/snapshots/7272eec02cb24bd6edd87a03cf0aa7c1d8cce68d"
answer_path="opt-smoothquant-13b"
torchrun --standalone --nproc_per_node=$device_number generate_answers.py \
    --model 'a' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 4 \
    --answer_path $answer_path \
    --max_length 1024

# 13b
model_path="/home/lczxl/.cache/huggingface/hub/models--mit-han-lab--opt-30b-smoothquant/snapshots/0b59c759b7e37558fb5e74176774b6eb7fa78474"
answer_path="opt-smoothquant-30b"
torchrun --standalone --nproc_per_node=$device_number generate_answers.py \
    --model 'a' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 2 \
    --answer_path $answer_path \
    --max_length 1024