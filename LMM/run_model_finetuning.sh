
#!/bin/bash

#SBATCH -J LMM_trainings
#SBATCH -o logs/resnet_hirise_landmark_new_2.out                       # STDOUT (%j = JobId)
#SBATCH -e logs/resnet_hirise_landmark_new_2.err                       # STDERR (%j = JobId)
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH -t 72:00:00

#SBATCH -p gpu
#SBATCH -G 1

##SBATCH -A mpurohit                                      # Account hours will be pulled from (commented out with double # in front)
#SBATCH --mail-type=ALL                                   # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=Mirali.V.Purohit@jpl.nasa.gov         # send-to address


cd /scratch-edge/large-mars-model/SURP_code/LMM


##### ----------------------------------------------------- Fine^2tuning ------------------------------------------------------------------------

#### ---------------- ResNet34 -------------------

### -------- DoMars16 -----------------

# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Single instrument '''
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Multiple instruments '''
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Last epoch '''
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_ctx_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_ctx_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Equal val loss epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_11_ctx_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_11_themis_resnet34_17.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_resnet34_500_themis_resnet34_17.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_11_ctx_resnet34_500_themis_resnet34_17.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

## ''' Early stopping epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_151_ctx_resnet34_289.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_151_themis_resnet34_193.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_resnet34_289_themis_resnet34_193.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_151_ctx_resnet34_289_themis_resnet34_193.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

## ''' Customized model combos '''
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_resnet34_11_ctx_resnet34_500_themis_resnet34_17_mc1.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_resnet34_11_ctx_resnet34_500_themis_resnet34_17_mc2.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_try/hirise_ctx_themis_resnet34_all_3.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

### -------- HiRISE Landmark ----------

# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Single instrument '''
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Multiple instruments '''
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Last epoch '''
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_ctx_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_ctx_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Equal val loss epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_11_ctx_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_11_themis_resnet34_17.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_resnet34_500_themis_resnet34_17.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_11_ctx_resnet34_500_themis_resnet34_17.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

## ''' Early stopping epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_151_ctx_resnet34_289.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_151_themis_resnet34_193.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_resnet34_289_themis_resnet34_193.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_151_ctx_resnet34_289_themis_resnet34_193.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

## ''' Customized model combos '''
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_resnet34_11_ctx_resnet34_500_themis_resnet34_17_mc1.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_resnet34_11_ctx_resnet34_500_themis_resnet34_17_mc2.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done
for i in {1..5}; do
python main_finetune.py --dataset hirise_landmark --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_try/hirise_ctx_themis_resnet34_all_3.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
done

### -------- Martian Frost -----------

# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_ctx_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_ctx_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

### ---- Atmospheric Dust EDR ----------

# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_ctx_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_ctx_resnet34_500_themis_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

### ---- Atmospheric Dust RDR ----------

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/resnet34/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_500_ctx_resnet34_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_11_themis_resnet34_17.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_resnet34_500_themis_resnet34_17.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model resnet34 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_resnet34_11_ctx_resnet34_500_themis_resnet34_17.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100


#### ------------- EfficientNet -----------------

### -------- DoMars16 -----------------

# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Single instrument '''
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Multiple instruments '''
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Last epoch '''
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_ctx_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Equal val loss epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_6_ctx_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_6_themis_efficientnet-v2-m_29.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_29.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_6_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_29.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# done

## ''' Early stopping epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_126_ctx_efficientnet-v2-m_463.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_126_themis_efficientnet-v2-m_285.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_efficientnet-v2-m_463_themis_efficientnet-v2-m_285.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_126_ctx_efficientnet-v2-m_463_themis_efficientnet-v2-m_285.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# done

## ''' Customized model combos '''
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_efficientnet-v2-m_6_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_29_mc1.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# done
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_efficientnet-v2-m_6_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_29_mc2.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# done
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_try/hirise_ctx_themis_efficientnet-v2-m_all_3.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# done

### -------- HiRISE Landmark ----------

# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Single instrument '''
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Multiple instruments '''
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Last epoch '''
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_ctx_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Equal val loss epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_6_ctx_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_6_themis_efficientnet-v2-m_29.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_29.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_6_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_29.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

## ''' Early stopping epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_126_ctx_efficientnet-v2-m_463.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_126_themis_efficientnet-v2-m_285.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_efficientnet-v2-m_463_themis_efficientnet-v2-m_285.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_126_ctx_efficientnet-v2-m_463_themis_efficientnet-v2-m_285.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

## ''' Customized model combos '''
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_efficientnet-v2-m_6_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_29_mc1.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_efficientnet-v2-m_6_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_29_mc2.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_try/hirise_ctx_themis_efficientnet-v2-m_all_3.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

### -------- Martian Frost -----------

# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128

# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128

# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128

# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_ctx_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128

### ---- Atmospheric Dust EDR ----------

# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_ctx_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

### ---- Atmospheric Dust RDR ----------

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/efficientnet-v2-m/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_ctx_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model efficientnet-v2-m --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_efficientnet-v2-m_500_ctx_efficientnet-v2-m_500_themis_efficientnet-v2-m_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100


#### ------------- ViTB_16 -----------------

### -------- DoMars16 -----------------

# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100ß
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Single instrument '''
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Multiple instruments '''
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Last epoch '''
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_ctx_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_ctx_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

## ''' Equal val loss epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_9_ctx_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_9_themis_vit-b-16_171.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_vit-b-16_500_themis_vit-b-16_171.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_9_ctx_vit-b-16_500_themis_vit-b-16_171.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# done

## ''' Early stopping epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_77_ctx_vit-b-16_160.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_77_themis_vit-b-16_46.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_vit-b-16_160_themis_vit-b-16_46.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_77_ctx_vit-b-16_160_themis_vit-b-16_46.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# done

## ''' Customized model combos '''
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_vit-b-16_9_ctx_vit-b-16_500_themis_vit-b-16_171_mc1.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done
# for i in {1..5}; do
# python main_finetune.py --dataset domars16 --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_vit-b-16_9_ctx_vit-b-16_500_themis_vit-b-16_171_mc2.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

### -------- HiRISE Landmark ----------

# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Single instrument '''
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Multiple instruments '''
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Last epoch '''
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_ctx_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_ctx_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

## ''' Equal val loss epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_9_ctx_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_9_themis_vit-b-16_171.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_vit-b-16_500_themis_vit-b-16_171.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_9_ctx_vit-b-16_500_themis_vit-b-16_171.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

## ''' Early stopping epoch '''
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_77_ctx_vit-b-16_160.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_77_themis_vit-b-16_46.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_vit-b-16_160_themis_vit-b-16_46.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_77_ctx_vit-b-16_160_themis_vit-b-16_46.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

## ''' Customized model combos '''
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_vit-b-16_9_ctx_vit-b-16_500_themis_vit-b-16_171_mc1.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done
# for i in {1..5}; do
# python main_finetune.py --dataset hirise_landmark --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/customized_models_pruning/hirise_vit-b-16_9_ctx_vit-b-16_500_themis_vit-b-16_171_mc2.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# done

### -------- Martian Frost -----------

# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128

# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128

# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128

# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_ctx_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128
# python main_finetune.py --dataset martian_frost --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_ctx_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --batch_size 128

### ---- Atmospheric Dust EDR ----------

# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_ctx_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_ctx_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

### ---- Atmospheric Dust RDR ----------

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/vit-b-16/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_ctx_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model vit-b-16 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_vit-b-16_500_ctx_vit-b-16_500_themis_vit-b-16_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100



#### ------------- SqueezeNet1.1 -----------------

### -------- DoMars16 -----------------

# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_ctx_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset domars16 --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_ctx_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

### ---- Atmospheric Dust EDR ----------

# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_ctx_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_edr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_ctx_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

### ---- Atmospheric Dust RDR ----------

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_ctx_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset atmospheric_dust_rdr --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_ctx_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

### -------- HiRISE Landmark ----------

# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_ctx_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data
# python main_finetune.py --dataset hirise_landmark --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_ctx_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100 --balance_data

### -------- Martian Frost -----------

# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining scratch_training --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining imagenet_pretrained --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/themis_ctx/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/pretraining/squeezenet1-1/hirise_ctx_themis/encoder_epoch_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_ctx_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/ctx_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100
# python main_finetune.py --dataset martian_frost --train_model squeezenet1-1 --which_pretraining finetuning --encoder_checkpoint /scratch-edge/large-mars-model/models/task_arithmetic/combined_models/hirise_squeezenet1-1_500_ctx_squeezenet1-1_500_themis_squeezenet1-1_500.pth --output_dir /scratch-edge/large-mars-model/models/task_arithmetic --wandb_enabled --num_epochs 100

