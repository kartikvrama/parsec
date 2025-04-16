# Training and Evaluation

Training scripts will save a checkpoint under `logs/<save_tag>`. Evaluation scripts will save model predictions and evaluation results under `results/<model_tag>`. For methods reqruing GPT-4, please set an environment variable called OPENAI_API_KEY with your personal OpenAI key.

Note:
* CFFM and NeatNet are trained and evaluated on individual environment instances, because of which they are excluded from the NovelEnvCategory experiment.
* LLM-based methods do not use any examples from the PARSEC dataset, and the evaluation for KnownEnv and NovelEnvCategory is identical.

## ContextSortLM

Training and Evaluation are combined in a single script:
```
python run_contextsortlm.py \
    --fold /path/to/fold \
    --dataset /path/to/dataset \
    --destination /path/to/destination
```

## APRICOT-NonInteractive
Generate APRICOT responses:
```
python run_apricot_noquery.py \
    --fold /path/to/fold \
    --dataset /path/to/dataset \
    --destination /path/to/destination
```

Evaluation:
```
python evaluate_apricot_noquery.py \
    --fold /path/to/fold \
    --dataset /path/to/dataset \
    --responses /path/to/apricot/responses \
    --model_tag <experiment-name>-apricot-noninteractive

```

## TidyBot-Random
Generate prompts offline:
```
python tidybot_core/save_llm_prompts.py \
    --fold /path/to/fold \
    --dataset /path/to/dataset \
    --destination_folder /path/to/destination
```

Generate TidyBot responses:
```
python run_tidybot.py
    --prompt_directory /path/to/prompt
    --destination_folder /path/to/response/destination
```

Evaluation:
```
python evaluate_tidybot.py \
    --fold /path/to/fold \
    --dataset /path/to/dataset \
    --responses /path/to/responses \
    --model_tag <experiment_name>-tidybot-random
```

## ConSOR

Script for generating semantic object embeddings:
```
python save_object_embedding.py \
    --model all-MiniLM-L6-v2
    --destination /path/to/destination
```
You can switch the text embedding model with other models in the `sentence_transformers` package.

Training:
```
python train_consor.py \
    --fold /path/to/fold \
    --dataset /path/to/dataset \
    --embedding /path/to/embedding/file
    --num_epochs 1500 \
    --wandb true \
    --lrate 1e-4 \
    --batch_size 2 \
    --hidden_layer_size 256
    --save_tag <experiment_name>-consor \
```

Evaluation:
```
python evaluate_consor.py \
    --fold /path/to/fold \
    --dataset /path/to/dataset \
    --checkpoint_folder /path/to/model/checkpoint \
    --stopping_metric edit_distance
    --model_tag <experiment_name>-consor \
```


## CF

Training:
```
python train_CF.py  \
    --fold /path/to/fold \
    --dataset /path/to/dataset \
    --learning_rate 1e-1 \
    --hidden_dimension 6 \
    --lambda_reg 1e-2 \
    --save_tag <experiment_name>-cf
```

Evaluation:
```
python evaluate_cf.py \
    --fold /path/to/fold \
    --dataset /path/to/dataset \
    --dataset $DATA_HOME/batch-2024-09-10/unseen_cat \
    --checkpoint_folder /path/to/model/checkpoint \
    --model_tag <experiment_name>-cf
```

## CFFM

Training: 
```
for ENV in {fridge,bathroom,bookshelf,dresser,kitchen}
do
  for VAR in {1,2,3}
  do
  python train_cfplus.py \
  --fold /path/to/fold \
  --dataset /path/to/dataset \
  --environment_cat $ENV \
  --environment_var $VAR \
  --hidden_dimension_cf 6 \
  --lambda_reg_cf 1e-2 \
  --learning_rate_cf 1e-1 \
  --hidden_dimension_fm 15 \
  --num_iter_fm 1000 \
  --init_lr_fm 0.03 \
  --init_stdev_fm 0.1 \
  --save_tag <experiment_tag>-cffm
  done
done
```

Evaluation:
```
for ENV in {fridge,bathroom,bookshelf,dresser,kitchen}
do
  for VAR in {1,2,3}
  do
  python evaluate_cfplus.py \
  --fold /path/to/fold \
  --dataset /path/to/dataset \
  --environment_cat $ENV \
  --environment_var $VAR \
  --checkpoint_folder /path/to/model/checkpoint \
  --model_tag <experiment_name>-cffm
  done
done
```

## NeatNet

  Training:
  ```
  for ENV in {fridge,bathroom,bookshelf,dresser,kitchen}
    do
      for VAR in {1,2,3}
      do
        python train_neatnet.py \
            --environment_cat $ENV \
            --environment_var $VAR \
            --fold /path/to/fold \
            --dataset /path/to/dataset \
            --num_epochs 2000 \
            --init_lr 1e-2 \
            --noise_scale 0.02 \
            --wandb true  \
            --batch_size 1 \
            --user_data_dir /path/to/user/data \
            --user_dim 2
            # save_tag is the same as model_tag
            --save_tag <experiment_name>-neatnet \ 
      done
    done
  ```

  Evaluation:
  ```
  for ENV in {fridge,bathroom,bookshelf,dresser,kitchen}
  do
    for VAR in {1,2,3}
    do
      python evaluate_neatnet.py \
        --device skynet \
        --partition overcap \
        --environment_cat $ENV \
        --environment_var $VAR \
        --fold /path/to/fold \
        --dataset /path/to/dataset \
        --stopping_metric edit_distance \
        --checkpoint_folder /path/to/model/checkpoint  \
        --model_tag <experiment_name>-neatnet
    done
  done
  ```
