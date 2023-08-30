DATA_DIR=dataset

MODEL_NAME=HAKE
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
TEST_SAMPLER_CLASS=TestSampler1
MAX_EPOCHS=1000
EMB_DIM=1000
LOSS=Adv_Loss
ADV_TEMP=1.0
TRAIN_BS=1024
EVAL_BS=16
NUM_NEG=1024
MARGIN=9.0
LR=5e-5

EARLY_STOP_PATIENCE=20
CHECK_PER_EPOCH=30
PHASE_WEIGHT=1.0
MODULUS_WEIGHT=3.5
NUM_WORKERS=40
GPU=0
LAMBDA_SYM=-2
LAMBDA_INV=-1
LAMBDA_SUB=-3
LAMBDA_COMP2=0.1


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
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --phase_weight $PHASE_WEIGHT \
    --modulus_weight $MODULUS_WEIGHT \
    --num_workers $NUM_WORKERS \
    --use_weight \
    --save_config \
    --test_only \
    --checkpoint_dir output/link_prediction/FB15K237/HAKE/epoch=209-Eval_mrr=0.347.ckpt \
    --use_comp2_weight \
    --lambda_comp2  $LAMBDA_COMP2 \
    # --use_sym_weight \
    # --lambda_sym  $LAMBDA_SYM \
    
    # --use_sub_weight \
    # --lambda_sub  $LAMBDA_SUB \
    # --use_inv_weight \
    # --lambda_inv  $LAMBDA_INV \
    
    # --use_comp3_weight \
    # --lambda_comp3  $LAMBDA_COMP3 \
    
    
    
    
    

    
