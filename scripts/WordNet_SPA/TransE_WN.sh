DATA_DIR=dataset

MODEL_NAME=TransE
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
TEST_SAMPLER_CLASS=TestSampler1
MAX_EPOCHS=1000
EMB_DIM=1024
LOSS=Adv_Loss
ADV_TEMP=1.0
TRAIN_BS=2048
EVAL_BS=16
NUM_NEG=256
MARGIN=6.0
LR=1e-5
CHECK_PER_EPOCH=50
NUM_WORKERS=16
GPU=0
LAMBDA_SYM=50


CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --litmodel_name $LITMODEL_NAME \
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
    --num_workers $NUM_WORKERS \
    --save_config \
    --test_only \
    --checkpoint_dir output/link_prediction/WN18RR/TransE/epoch=749-Eval_mrr=0.218.ckpt \
    --use_sym_weight \
    --lambda_sym  $LAMBDA_SYM
    







