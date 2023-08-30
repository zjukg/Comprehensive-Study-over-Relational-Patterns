DATA_DIR=dataset

MODEL_NAME=DistMult
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
TEST_SAMPLER_CLASS=TestSampler1
MAX_EPOCHS=8000
EMB_DIM=2048
LOSS=Adv_Loss
TRAIN_BS=2048
EVAL_BS=16
NUM_NEG=128
MARGIN=200
LR=3e-6
REGULARIZATION=1e-7
CHECK_PER_EPOCH=500
NUM_WORKERS=16
GPU=0
LAMBDA_SYM=-2
LAMBDA_INV=-2
LAMBDA_SUB=-4
LAMBDA_COMP2=0.00001

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss $LOSS \
    --negative_adversarial_sampling \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LR \
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --use_weight \
    --save_config \
    --test_only \
    --checkpoint_dir output/link_prediction/FB15K237/DistMult/epoch=3499-Eval_mrr=0.259.ckpt \
    --use_sym_weight \
    --lambda_sym  $LAMBDA_SYM \
    # --use_comp2_weight \
    # --lambda_comp2  $LAMBDA_COMP2 \
    # --use_sub_weight \
    # --lambda_sub  $LAMBDA_SUB \
    # --use_inv_weight \
    # --lambda_inv  $LAMBDA_INV \
    
    
    
    
    
    
    
    # --use_comp3_weight \
    # --lambda_comp3  $LAMBDA_COMP3 \