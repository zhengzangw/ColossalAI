set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

device_number=1
model_name="opt"
model_path="/home/lczxl/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0"
dataset="/home/lcxyc/data1/evaluate/ColossalAI/applications/Chat/evaluate/table/evaluation_dataset/cn/evaluate_cn_v4_300.json"
answer_path="answer"

set_n_least_used_CUDA_VISIBLE_DEVICES $device_number

torchrun --standalone --nproc_per_node=$device_number generate_answers.py \
    --model 'a' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 32 \
    --answer_path $answer_path \
    --max_length 1024
    # --max_datasets_size 80 \
