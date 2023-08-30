DATA_DIR=dataset
MODEL_NAME=DualE
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
TRAIN_SAMPLER_CLASS=BernSampler
TEST_SAMPLER_CLASS=TestSampler1
MAX_EPOCHS=20000
EMB_DIM=200
LOSS=Softplus_Loss
ADV_TEMP=1.0
EVAL_BS=16
NUM_NEG=2
OPTIM=Adagrad
MARGIN=1.0
LR=0.022
CHECK_PER_EPOCH=100
NUM_WORKERS=8
GPU=0
PATIENCE=10
NUM_BATCHES=10
REGULARIZATION=0.25
REGULARIZATION_TWO=0.25
WEIGHT_IN_TRAIN_VALID=1
LAMBDA_SYM=2

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss $LOSS \
    --adv_temp $ADV_TEMP \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --optim_name $OPTIM \
    --margin $MARGIN \
    --lr $LR \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --save_config \
    --early_stop_patience $PATIENCE \
    --num_batches $NUM_BATCHES \
    --regularization $REGULARIZATION \
    --regularization_two $REGULARIZATION_TWO \
    --test_only \
    --checkpoint_dir output/link_prediction/WN18RR/DualE/epoch=4299-Eval_mrr=0.494.ckpt \
    --use_sym_weight \
    --lambda_sym  $LAMBDA_SYM

