DATA_DIR=dataset

MODEL_NAME=ComplEx
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
TEST_SAMPLER_CLASS=TestSampler1
MAX_EPOCHS=5000
EMB_DIM=500
LOSS=Adv_Loss
ADV_TEMP=1.0
TRAIN_BS=1024
EVAL_BS=16
NUM_NEG=64
MARGIN=12.0
LR=1e-5
REGULARIZATION=1e-7
CHECK_PER_EPOCH=50
NUM_WORKERS=16
GPU=0
LAMBDA_SYM=-2
LAMBDA_INV=-1
LAMBDA_SUB=-4
LAMBDA_COMP2=-0.01


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
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --save_config \
    --test_only \
    --checkpoint_dir output/link_prediction/FB15K237/ComplEx/epoch=199-Eval_mrr=0.262.ckpt \
    # --use_comp2_weight \
    # --lambda_comp2  $LAMBDA_COMP2 \
    # --use_sub_weight \
    # --lambda_sub  $LAMBDA_SUB \
    # --use_inv_weight \
    # --lambda_inv  $LAMBDA_INV \
    # --use_sym_weight \
    # --lambda_sym  $LAMBDA_SYM \
    
    
    
    

    
    
    
    
    

    








