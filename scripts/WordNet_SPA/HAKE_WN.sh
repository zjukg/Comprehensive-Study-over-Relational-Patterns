DATA_DIR=dataset

MODEL_NAME=HAKE
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
TEST_SAMPLER_CLASS=TestSampler1
MAX_EPOCHS=1000
EMB_DIM=500
LOSS=Adv_Loss
ADV_TEMP=0.5
TRAIN_BS=256
EVAL_BS=8
NUM_NEG=1024
MARGIN=6.0
LR=5e-5
CHECK_PER_EPOCH=30
PHASE_WEIGHT=0.5
MODULUS_WEIGHT=0.5
NUM_WORKERS=40
GPU=0
LAMBDA_SYM=1

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss $LOSS \
    --adv_temp $ADV_TEMP \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LR \
    --check_per_epoch $CHECK_PER_EPOCH \
    --phase_weight $PHASE_WEIGHT \
    --modulus_weight $MODULUS_WEIGHT \
    --num_workers $NUM_WORKERS \
    --save_config \
    --test_only \
    --checkpoint_dir output/link_prediction/WN18RR/HAKE/epoch=119-Eval_mrr=0.489.ckpt \
    --use_sym_weight \
    --lambda_sym  $LAMBDA_SYM