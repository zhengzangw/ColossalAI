export CUDA_VISIBLE_DEVICES="7"
device_number=1
model_name="opt-bitsandbytes"
model_path="/home/lczxl/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0"
dataset="/home/lcxyc/data1/evaluate/ColossalAI/applications/Chat/evaluate/table/evaluation_dataset/cn/evaluate_cn_v4_300.json"
answer_path="opt-bitsandbytes-6.7b"


torchrun --standalone --nproc_per_node=$device_number generate_answers.py \
    --model 'a' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 8 \
    --answer_path $answer_path \
    --max_length 1024
