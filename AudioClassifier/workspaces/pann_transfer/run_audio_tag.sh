#!bin/bash

CHECKPOINT_PATH="./AudioClassifier/workspaces/pann_transfer/checkpoints/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/10000_iterations.pth"
AUDIO_PATH="./AudioClassifier/workspaces/pann_transfer/resources/AuSep_1_flute_40_Miserere.wav"
MODEL_TYPE="Transfer_Cnn14"

CUDA_VISIBLE_DEVICES=0 python3 AudioClassifier2.0/workspaces/pann_transfer/pytorch/inference.py audio_tagging --sample_rate=48000 --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path=$AUDIO_PATH --cuda

