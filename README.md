# Personalized Robotic Object Rearrangement from Scene Context

This is the official repository for the paper titled "Personalized Robotic Object Rearrangement from Scene Context", accepted at ROMAN 2025 ([arxiv](https://arxiv.org/abs/2505.11108)). The repo contains code for generating PARSEC evaluation examples from crowdsourced user data, implementations of existing personalized object rearrangement works, implementation our proposed approach ContextSortLM, and various scripts to evaluate and visualize model predictions.

[Installation.md](./docs/installation.md) contains instructions on installing dependencies.

## PARSEC Dataset
PARSEC contains a novel object rearrangement dataset of 110k evaluation examples, generated from crowdsourcing using the data collection interface [here](https://github.com/kartikvrama/parsec-dataset-and-evaluation/tree/main/data_collection_fullstack). We asked 72 online users to perform five household organizational tasks: stocking a kitchen pantry, arranging a bathroom cabinet, rearranging a bedroom dresser, stocking a fridge, and decorating a display shelf. In total, we collected 432 object arrangements, involving 93 household objects and spanning 72 users and 15 environment instances. You can download the dataset at this [dropbox link](https://www.dropbox.com/scl/fi/oeuq12h9x32pfwx6vv3h7/parsec_data.zip?rlkey=vc50ndzgxj3qjtwkndznzrmxw&st=h4z9dlul&dl=0).

[Dataset.md](./docs/dataset.md) contains details about the dataset and instructions on generating evaluation examples and visulalizing examples.

## Training and Evaluation Rearrangement Models
The CF, CFFM, NeatNet, and ConSOR methods are trained on PARSEC dataset, while the TidyBot-Random, APRICOT-NonInteractive, and ContextSortLM models are LLM-based models that do not use any examples from this dataset. Model checkpoints for all models trained on this dataset are available at this link: [coming soon!](TODO)

The detailed instructions for running each model can be found in [training_evaluation.md](docs/training_evaluation.md).

## Plotting Results

### Computational Experiments

The following script generates placement accuracy and error metrics. The script takes in the model results and the fold directory containing the ground truth data. The `--plot` flag generates plots for the results.
```
python compare_results.py \
    --cf /path/to/cf/results \
    --cffm /path/to/cff,/results \
    --consor /path/to/consor/results \
    --neatnet /path/to/neatnet/results \
    --tidybot_random /path/to/tidybot/results \
    --llm_summarizer /path/to/contextsortlm/results \
    --apricot  /path/to/apricot/results \
    --fold /path/to/fold \
    --plot
```

The script will generate the following plots:
- **Placement Accuracy**: This plot shows the placement accuracy of each model on the test set, grouped by environment category.
- **Error Metrics**: This plot shows the SED (Scene Edit Distance) and IGO (Number of Incorrectly Grouped Objects) of each model as a function of the number of pre-arranged objects in the initial environment state.

### Online Rater Evaluation
The code for collecting user arrangement data and evaluating model predictions with online raters is available at [parsec-dataset-and-evaluation repository](https://github.com/kartikvrama/parsec-dataset-and-evaluation).
