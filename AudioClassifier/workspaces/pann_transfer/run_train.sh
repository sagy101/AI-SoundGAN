#!bin/bash


DATASET_DIR="./datasets/Sub-URMP"
WORKSPACE="./workspaces/pann_transfer"


python3 workspaces/pann_transfer/utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

PRETRAINED_CHECKPOINT_PATH="./released_models/Cnn14_mAP=0.431.pth"

CUDA_VISIBLE_DEVICES=1
python3 workspaces/pann_transfer/pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda

#####
#MODEL_TYPE="Transfer_Cnn13"
#PRETRAINED_CHECKPOINT_PATH="/vol/vssp/msos/qk/bytedance/workspaces_important/pub_audioset_tagging_cnn_transfer/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/660000_iterations.pth"
#pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --freeze_base --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --few_shots=10 --random_seed=1000 --resume_iteration=0 --stop_iteration=10000 --cuda

#python3 workspaces/panns_transfer_to_gtzan/utils/plot_statistics.py plot --workspace=$WORKSPACE --select=2_cnn13