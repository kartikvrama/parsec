#!/bin/sh
MYDIR="/media/kartik/disk-2/robo_declutter/results/2024-08-26/ood"
python stat_analysis_results.py \
    --cf $MYDIR/cf-OOD-2024-08-19/ \
    --cffm $MYDIR/cfplus-OOD-2024-08-19 \
    --consor $MYDIR/consor-OOD-2024-07-23-IV/ \
    --neatnet $MYDIR/neatnet-OOD-2024-08-15/ \
    --tidybot_random $MYDIR/tidybot-OOD-2024-08-23/ \
    --llm_summarizer $MYDIR/tidybot_plus-OOD-2024-08-26 \
    --fold $DATA_HOME/folds-2024-07-20/out_distribution.pkl