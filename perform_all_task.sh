#!/bin/bash
set -e

# Activate conda environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "afforddp" ]; then
    echo "Activating afforddp environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate afforddp
fi

# Default values
TASK_NAME="PullDrawer"
OBJ_ID=""
PART_ID=-1
SEED=42
CUDA_ID=0
NUM_PARALLEL=1
DATA_DIR="record"
SAVE_DIR="data"
CKPT_PATH=""

usage() {
    cat << EOF
Usage: $0 --task_name <collect_demon|process_data|train_policy|eval_policy|all> [OPTIONS]

Required:
  --task_name         Task: collect_demon, process_data, train_policy, eval_policy, all
  --config_name       Config name: PullDrawer.yaml or OpenDoor.yaml
  --obj_id            Object ID (required for collect_demon, eval_policy, all)

Optional:
  --part_id           Part ID (default: -1)
  --seed              Seed (default: 42)
  --cuda_id           GPU ID (default: 0)
  --num_parallel      Parallel workers (default: 1)
  --data_dir          Data dir (default: record)
  --save_dir          Save dir (default: data)
  --ckpt_path         Checkpoint path (required for eval_policy)

Examples:
  # Collect demonstrations with 4 parallel workers
  $0 --task_name collect_demon --config_name PullDrawer.yaml --obj_id 46859 --part_id 1 --num_parallel 4

  # Process collected data
  $0 --task_name process_data --config_name PullDrawer.yaml --obj_id 46859

  # Train policy with custom seed and GPU
  $0 --task_name train_policy --config_name OpenDoor.yaml --seed 42 --cuda_id 0

  # Evaluate trained policy on specific object
  $0 --task_name eval_policy --config_name PullDrawer.yaml --obj_id 46859 --ckpt_path outputs/2025.01.28/14.30.45_train_afford_cond_pointcloud_dp

  # Run complete workflow (collect, process, train, eval)
  $0 --task_name all --config_name OpenDoor.yaml --obj_id 46859 --part_id 1 --num_parallel 4
EOF
    exit 1
}

TASK=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_name) TASK="$2"; shift 2 ;;
        --config_name) CONFIG_NAME="$2"; shift 2 ;;
        --obj_id) OBJ_ID="$2"; shift 2 ;;
        --part_id) PART_ID="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --cuda_id) CUDA_ID="$2"; shift 2 ;;
        --num_parallel) NUM_PARALLEL="$2"; shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --save_dir) SAVE_DIR="$2"; shift 2 ;;
        --ckpt_path) CKPT_PATH="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validate
[ -z "$TASK" ] && { echo "Error: --task_name required"; usage; }
[ -z "$CONFIG_NAME" ] && { echo "Error: --config_name required (use PullDrawer.yaml or OpenDoor.yaml)"; usage; }
[[ "$TASK" == "collect_demon" || "$TASK" == "eval_policy" || "$TASK" == "all" ]] && [ -z "$OBJ_ID" ] && { echo "Error: --obj_id required"; usage; }
[[ "$TASK" == "eval_policy" ]] && [ -z "$CKPT_PATH" ] && { echo "Error: --ckpt_path required"; usage; }

TASK_NAME=$(echo "$CONFIG_NAME" | sed 's/.yaml//')

collect_demonstrations() {
    echo "==> Collecting demonstrations (obj_id=$OBJ_ID, part_id=$PART_ID, parallel=$NUM_PARALLEL)"
    python collect_demonstrations.py \
        --config_name "$CONFIG_NAME" \
        --save_dir "$DATA_DIR" \
        --obj_id "$OBJ_ID" \
        --part_id "$PART_ID" \
        --seed "$SEED" \
        --num_parallel "$NUM_PARALLEL"
    echo "✓ Done"
}

process_data() {
    [ -n "$OBJ_ID" ] && PROCESS_DATA_DIR="$DATA_DIR/$TASK_NAME/$OBJ_ID" || PROCESS_DATA_DIR="$DATA_DIR/$TASK_NAME"
    echo "==> Processing data ($PROCESS_DATA_DIR -> $SAVE_DIR)"
    python process_data.py --data_dir "$PROCESS_DATA_DIR" --save_dir "$SAVE_DIR"
    echo "✓ Done"
}

train_policy() {
    echo "==> Training policy (seed=$SEED, cuda=$CUDA_ID)"
    bash train.sh "$SEED" "$CUDA_ID"
    echo "✓ Done"
}

eval_policy() {
    echo "==> Evaluating policy (ckpt=$CKPT_PATH, obj_id=$OBJ_ID)"
    [ ! -d "$CKPT_PATH" ] && { echo "Error: Checkpoint not found: $CKPT_PATH"; exit 1; }
    bash eval.sh "$CKPT_PATH" "$OBJ_ID"
    echo "✓ Done"
}

echo "AffordDP Workflow - Task: $TASK"

case $TASK in
    collect_demon) collect_demonstrations ;;
    process_data) process_data ;;
    train_policy) train_policy ;;
    eval_policy) eval_policy ;;
    all)
        echo "Running complete workflow..."
        collect_demonstrations
        process_data
        train_policy

        LATEST_CKPT=$(find outputs -type d -name "checkpoints" 2>/dev/null | head -n 1 | sed 's|/checkpoints||')
        if [ -z "$LATEST_CKPT" ]; then
            echo "Warning: No checkpoint found. Run evaluation manually:"
            echo "  $0 --task_name eval_policy --obj_id $OBJ_ID --ckpt_path <path>"
        else
            echo "Found checkpoint: $LATEST_CKPT"
            CKPT_PATH="$LATEST_CKPT"
            eval_policy
        fi
        echo "All tasks completed!"
        ;;
    *)
        echo "Error: Invalid task '$TASK'"
        usage
        ;;
esac
