# Dataset Generation and Splitting

## Dataset Generation

To generate evaluation examples from user object arrangements, run the following script from the `data_permutation` folder:
```
python permute_user_data.py \
    --data /path/to/user/data \
    --destination /path/to/destination \
    --verbose True
```
The `dryrun` flag calculates the approximate number of examples per environmennt instance.

## K-fold Cross Validation Sets

We run two computational experiments using k-fold cross-validation -- KnownEnv and NovelEnvCategory.

To generate data folds for KnownEnv, run the following from the `data_split` folder:
```
python split_data.py \
    --user_data_dir /path/to/user/data \
    --dataset /path/to/permuted/data \
    --user_list ../labels/eligible_users.txt \
    --destination /path/to/destination \
    --in_distribution
```

To generate data folds for the NovelEnvCategory, run the following from the `data_split` folder:
```
python split_data.py \
    --user_data_dir /path/to/user/data \
    --dataset /path/to/permuted/data \
    --user_list ../labels/eligible_users.txt \
    --destination /path/to/destination \
    --unseen_cat
```

`split_data.py` generates a pickle file containing the evaluation example IDs per fold for training, validation, and test. This file needs to be converted into an actual dataset file, which is obtained by running the following script from the `data_split` folder:
```
python save_fold_pkl.py \
    --dataset /path/to/permuted/data \
    --fold /path/to/fold/file \
    --destination /path/to/destination
``` 

## Visualizing Evaluation Examples
Coming soon!