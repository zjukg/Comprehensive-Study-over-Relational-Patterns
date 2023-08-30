DATA_DIR=dataset
MODEL_NAME=DualE
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
TRAIN_SAMPLER_CLASS=BernSampler
TEST_SAMPLER_CLASS=TestSampler1
MAX_EPOCHS=5000
EMB_DIM=100
LOSS=Softplus_Loss
ADV_TEMP=1.0
EVAL_BS=16
NUM_NEG=10
OPTIM=Adagrad
MARGIN=1.0
LR=0.02
CHECK_PER_EPOCH=100
NUM_WORKERS=8
GPU=0
PATIENCE=10
NUM_BATCHES=10
REGULARIZATION=0.1
REGULARIZATION_TWO=0.1
LAMBDA_SYM=-2
LAMBDA_INV=-1
LAMBDA_SUB=-3
LAMBDA_COMP2=-0.01


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
    --checkpoint_dir output/link_prediction/FB15K237/DualE/epoch=2499-Eval_mrr=0.338.ckpt \
    --use_sym_weight \
    --lambda_sym  $LAMBDA_SYM \
    # --use_sub_weight \
    # --lambda_sub  $LAMBDA_SUB \
    # --use_inv_weight \
    # --lambda_inv  $LAMBDA_INV \
    
    # --use_comp2_weight \
    # --lambda_comp2  $LAMBDA_COMP2 \
    
    
    
    
    
    # --use_comp3_weight \
    # --lambda_comp3  $LAMBDA_COMP3 \
    
    
    
    
    
    
    
    
    
    
    
    
    

    