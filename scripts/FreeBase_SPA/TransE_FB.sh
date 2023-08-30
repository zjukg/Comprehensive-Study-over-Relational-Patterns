DATA_DIR=dataset

MODEL_NAME=TransE
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
TRAIN_SAMPLER_CLASS=UniSampler #BernSampler1  UniSampler
TEST_SAMPLER_CLASS=TestSampler1
MAX_EPOCHS=5000
EMB_DIM=1000
LOSS=Adv_Loss
ADV_TEMP=1.0
TRAIN_BS=1024
EVAL_BS=16
NUM_NEG=256
MARGIN=9.0
LR=1e-6
CHECK_PER_EPOCH=100
NUM_WORKERS=16
GPU=0
LAMBDA_SYM=-2
LAMBDA_INV=-2
LAMBDA_SUB=-3
LAMBDA_COMP2=0.2

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --data_path $DATA_PATH \
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
    --checkpoint_dir output/link_prediction/FB15K237/TransE/epoch=2099-Eval_mrr=0.357.ckpt \
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
    
    
   
    
    
    


    