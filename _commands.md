  # Neatnet
  ## Neatnet training with out-of-distribution preferences
  Train:
  ```
  for ENV in {fridge,bathroom,bookshelf,dresser,kitchen}
    do
      for VAR in {1,2,3}
      do
        sbatch train_neatnet.sh --device skynet --partition overcap --environment_cat $ENV --environment_var $VAR --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl --dataset $DATA_HOME/batch-2024-07-20/out_distribution --save_tag UOBJ-$ENV-$VAR-2024-08-06 --num_epochs 2000 --init_lr 1e-2 --noise_scale 0.02 --wandb true  --batch_size 1 --user_data_dir $DATA_HOME/arrangements_json --user_dim 2
      done
    done
  ```

  Eval:
  ```
  for ENV in {fridge,bathroom,bookshelf,dresser,kitchen}
  do
    for VAR in {1,2,3}
    do
      sbatch evaluate_neatnet.sh --device skynet --partition overcap --environment_cat $ENV --environment_var $VAR --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl --dataset $DATA_HOME/batch-2024-07-20/out_distribution --stopping_metric edit_distance --checkpoint_folder $PROJ_HOME/logs/neatnet-OOD-$ENV-$VAR-2024-07-22/  --model_tag OOD-2024-08-15
    done
  done
  ```

  ## Neatnet training unseen objects fold
  Train:
  ```
  for ENV in {fridge,bathroom,bookshelf,dresser,kitchen}
    do
      for VAR in {1,2,3}
      do
        sbatch train_neatnet.sh --device skynet --partition overcap --environment_cat $ENV --environment_var $VAR --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj --save_tag UOBJ-$ENV-$VAR-2024-08-08-II --num_epochs 2000 --init_lr 1e-2 --noise_scale 0.02 --wandb true  --batch_size 1 --user_data_dir $DATA_HOME/arrangements_json --user_dim 2
      done
    done
  ```

  Eval:
  ```
  for ENV in {fridge,bathroom,bookshelf,dresser,kitchen}
  do
    for VAR in {1,2,3}
    do
      sbatch evaluate_neatnet.sh --device skynet --partition overcap --user_data_dir $DATA_HOME/arrangements_json --environment_cat $ENV --environment_var $VAR --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj --stopping_metric edit_distance --checkpoint_folder $PROJ_HOME/logs/neatnet-UOBJ-$ENV-$VAR-2024-08-08-II/ --model_tag UOBJ-2024-08-08-II 
    done
  done
  ```

  DEBUG:
  ```
  bash train_neatnet.sh --environment_cat kitchen --environment_var 2 --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj --save_tag DUMMY --num_epochs 10 --init_lr 1e-2 --noise_scale 0.02 --batch_size 1 --user_data_dir $DATA_HOME/arrangements_json
  ```

  # ConSOR
  ## ConSOR training with out-of-distribution preferences
  Train:
  ```
  sbatch train_consor.sh --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl --dataset $DATA_HOME/batch-2024-07-20/out_distribution --save_tag OOD-2024-07-23-IV --num_epochs 2000 --device skynet --wandb true --lrate 1e-4 --batch_size 2 --hidden_layer_size 256
  ```

  Eval:
  ```
  sbatch evaluate_consor.sh --device skynet --partition rail-lab --dataset $DATA_HOME/batch-2024-07-20/out_distribution --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl --model_tag OOD-2024-07-23-IV --checkpoint_folder $PROJ_HOME/logs/consor-OOD-2024-07-23-IV/ --stopping_metric edit_distance
  ```
  ## ConSOR training unseen objects fold
  Train:
  ```
  sbatch train_consor.sh --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj --save_tag UOBJ-2024-08-06 --num_epochs 1500 --device skynet --wandb true --lrate 1e-4 --batch_size 2 --hidden_layer_size 256
  ```

  Eval:
  ```
  sbatch evaluate_consor.sh --device skynet --partition rail-lab --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj --model_tag UOBJ-2024-08-06 --checkpoint_folder $PROJ_HOME/logs/consor-UOBJ-2024-08-06/ --stopping_metric edit_distance
  ```

  ## ConSOR training with unseen env fold
  Train:
  ```
  sbatch train_consor.sh --fold $DATA_HOME/folds-2024-07-20/unseen_env.pkl --dataset $DATA_HOME/batch-2024-07-20/unseen_env --save_tag UENV-2024-08-08 --num_epochs 1500 --device skynet --wandb true --lrate 1e-4 --batch_size 2 --hidden_layer_size 256
  ```

  Eval:
  ```
  sbatch evaluate_consor.sh --device skynet --partition rail-lab --fold $DATA_HOME/folds-2024-07-20/unseen_env.pkl --dataset $DATA_HOME/batch-2024-07-20/unseen_env --model_tag UENV-2024-08-08 --checkpoint_folder $PROJ_HOME/logs/consor-UENV-2024-08-08/ --stopping_metric edit_distance
  ```

  ## ConSOR training with unseen env-cat fold
  Train:
  ```
  sbatch train_consor.sh --fold $DATA_HOME/folds-2024-09-10/unseen_cat.pkl --dataset $DATA_HOME/batch-2024-09-10/unseen_cat --save_tag UENVCAT-2024-09-10 --num_epochs 1500 --device skynet --wandb true --lrate 1e-4 --batch_size 2 --hidden_layer_size 256
  ```

  Eval:
  ```
  sbatch evaluate_consor.sh --device skynet --partition overcap --fold $DATA_HOME/folds-2024-09-10/unseen_cat.pkl --dataset $DATA_HOME/batch-2024-09-10/unseen_cat --model_tag UENVCAT-2024-09-10 --checkpoint_folder $PROJ_HOME/logs/consor-UENVCAT-2024-09-10/ --stopping_metric edit_distance
  ```

  # Declutter
  ## Declutter training with out-of-distribution preferences
  Train:
  ```
  sbatch train_declutter.sh --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl --dataset $DATA_HOME/batch-2024-07-20/out_distribution --save_tag OOD-2024-08-30 --num_epochs 1000 --device skynet --wandb true --lrate 1e-4 --lr_scheduler_tmax 500 --alpha 1.0  --beta 0.0
  ```

  Eval:
  ```
  sbatch evaluate_declutter.sh --device skynet --partition rail-lab   --dataset $DATA_HOME/batch-2024-07-20/out_distribution --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl --model_tag OOD-2024-08-31 --checkpoint_folder $PROJ_HOME/logs/declutter-OOD-2024-08-31/ --stopping_metric edit_distance
  ```

  DEBUG:
  ```
  sbatch train_declutter.sh --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl --dataset $DATA_HOME/batch-2024-07-20/out_distribution --save_tag OOD-2024-08-08-NEW --num_epochs 2000 --device skynet --wandb true --lrate 5e-4 --lr_scheduler_tmax 50 --alpha 1.0 --beta 0.0  
  ```

  ## Declutter training unseen objects fold
  Train:
  ```
  sbatch train_declutter.sh --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj --save_tag UOBJ-2024-08-08-I --num_epochs 2000 --device skynet --wandb true --lrate 1e-4 --lr_scheduler_tmax 50 --alpha 0.75 --beta 0.25  
  ```
  DEBUG
  ```
  bash train_declutter.sh --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj --save_tag DUMMY --num_epochs 2000 --device skynet --lrate 1e-4 --lr_scheduler_tmax 50 --alpha 0.75 --beta 0.25  
  ```
  
  Eval:
  ```
  sbatch evaluate_declutter.sh --device skynet --partition rail-lab --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj --model_tag UOBJ-2024-08-08-I --checkpoint_folder $PROJ_HOME/logs/declutter-UOBJ-2024-08-08-I/ --stopping_metric edit_distance
  ```

  ## Declutter training unseen env fold
  Train:
  ```
  sbatch train_declutter.sh --fold $DATA_HOME/folds-2024-07-20/unseen_env.pkl --dataset $DATA_HOME/batch-2024-07-20/unseen_env --save_tag UENV-2024-08-28 --num_epochs 1500 --device skynet --wandb true --lrate 1e-4 --lr_scheduler_tmax 500 --alpha 1.0 --beta 0.0
  ```

  Eval:
  ```
  sbatch evaluate_declutter.sh --device skynet --partition rail-lab   --fold $DATA_HOME/folds-2024-07-20/unseen_env.pkl --dataset $DATA_HOME/batch-2024-07-20/unseen_env --model_tag UENV-2024-08-31 --checkpoint_folder $PROJ_HOME/logs/declutter-UENV-2024-08-31/ --stopping_metric edit_distance
  ```

  # CF
  ## CF training with out-of-distribution preferences
  Train:
  ```
  sbatch train_cf.sh --device skynet --partition overcap --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl --dataset $DATA_HOME/batch-2024-07-20/out_distribution  --learning_rate 1e-1 --hidden_dimension 6 --lambda_reg 1e-2 --save_tag OOD-2024-08-19
  ```

  Eval:
  ```
  sbatch evaluate_cf.sh --device skynet --partition overcap  --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl --dataset $DATA_HOME/batch-2024-07-20/out_distribution --checkpoint_folder $PROJ_HOME/logs/cf-OOD-2024-08-19/ --model_tag OOD-2024-08-19
  ```

  ## CF training unseen objects fold
  Train:
  ```
  sbatch train_cf.sh --device skynet --partition rail-lab  --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj  --learning_rate 1e-1 --hidden_dimension 6 --lambda_reg 1e-2 --save_tag UOBJ-2024-08-08
  ```

  Eval:
  ```
  sbatch evaluate_cf.sh --device skynet --partition rail-lab  --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj --model_tag UOBJ-2024-08-08 --checkpoint_folder $PROJ_HOME/logs/cf-UOBJ-2024-08-08/
  ```

  ## CF training unseen env fold
  Train:
  ```
  sbatch train_cf.sh --device skynet --partition rail-lab  --fold $DATA_HOME/folds-2024-07-20/unseen_env.pkl --dataset $DATA_HOME/batch-2024-07-20/unseen_env --learning_rate 1e-1 --hidden_dimension 6 --lambda_reg 1e-2 --save_tag UENV-2024-08-19
  ```

  Eval:
  ```
  sbatch evaluate_cf.sh --device skynet --partition rail-lab   --fold $DATA_HOME/folds-2024-07-20/unseen_env.pkl --dataset $DATA_HOME/batch-2024-07-20/unseen_env --checkpoint_folder $PROJ_HOME/logs/cf-UENV-2024-08-19/  --model_tag UENV-2024-08-19
  ```

## CF training unseen env-cat fold
  Train:
  ```
  sbatch train_cf.sh --device skynet --partition rail-lab  --fold $DATA_HOME/folds-2024-09-10/unseen_cat.pkl --dataset $DATA_HOME/batch-2024-09-10/unseen_cat --learning_rate 1e-1 --hidden_dimension 6 --lambda_reg 1e-2 --save_tag UENVCAT-2024-09-10
  ```

  Eval:
  ```
  sbatch evaluate_cf.sh --device skynet --partition rail-lab   --fold $DATA_HOME/folds-2024-09-10/unseen_cat.pkl --dataset $DATA_HOME/batch-2024-09-10/unseen_cat --checkpoint_folder $PROJ_HOME/logs/cf-UENVCAT-2024-09-10/  --model_tag UENVCAT-2024-09-10
  ```

# CFPlus
## CFPlus training with out-of-distribution preferences
Train:
```
for ENV in {fridge,bathroom,bookshelf,dresser,kitchen}
do
  for VAR in {1,2,3}
  do
  sbatch train_cfplus.sh \
  --device skynet \
  --partition overcap \
  --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl \
  --dataset $DATA_HOME/batch-2024-07-20/out_distribution \
  --environment_cat $ENV \
  --environment_var $VAR \
  --hidden_dimension_cf 6 \
  --lambda_reg_cf 1e-2 \
  --learning_rate_cf 1e-1 \
  --hidden_dimension_fm 15 \
  --num_iter_fm 1000 \
  --init_lr_fm 0.03 \
  --init_stdev_fm 0.1 \
  --save_tag OOD-$ENV-$VAR-2024-08-19
  done
done
```

Eval:
```
for ENV in {fridge,bathroom,bookshelf,dresser,kitchen}
do
  for VAR in {1,2,3}
  do
  sbatch evaluate_cfplus.sh \
  --device skynet \
  --partition overcap \
  --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl \
  --dataset $DATA_HOME/batch-2024-07-20/out_distribution \
  --environment_cat $ENV \
  --environment_var $VAR \
  --checkpoint_folder $PROJ_HOME/logs/cfplus-OOD-$ENV-$VAR-2024-08-19 \
  --model_tag OOD-2024-08-19
  done
done
```

# TidyBot

## TidyBot eval with out-of-distribution preferences
```
bash evaluate_tidybot.sh --device skynet --partition rail-lab --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl --dataset $DATA_HOME/batch-2024-07-20/out_distribution --responses results/tidybot_responses-gpt4 --model_tag OOD-2024-08-23
```
## TidyBot eval with unseen-object preferences
```
bash evaluate_tidybot.sh --device skynet --partition rail-lab --fold $DATA_HOME/folds-2024-08-07/unseen_obj.pkl --dataset $DATA_HOME/batch-2024-08-07/unseen_obj --responses results/tidybot_responses-gpt4 --model_tag UOBJ-2024-08-08
```
## TidyBot eval with unseen-env preferences
```
bash evaluate_tidybot.sh --device skynet --partition rail-lab --fold $DATA_HOME/folds-2024-07-20/unseen_env.pkl --dataset $DATA_HOME/batch-2024-07-20/unseen_env --responses results/tidybot_responses-gpt4 --model_tag UENV-2024-08-23
```
