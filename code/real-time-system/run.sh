gpu_n=$1
DATASET=$2

seed=5
BATCH_SIZE=32
SLIDE_WIN=5
dim=64
out_layer_num=1
SLIDE_STRIDE=1
significance_level=0.05
out_layer_inter_dim=128
num_folds=5
decay=0
visualize_graph=true


path_pattern="${DATASET}"
COMMENT="${DATASET}"
load_model_path="./pretrained/swat/best_09_23-11-00-17.pt"

EPOCH=30
report='best'

if [[ "$gpu_n" == "cpu" ]]; then
    python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -load_model_path $load_model_path \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -num_folds $num_folds \
        -report $report \
        -significance_level $significance_level \
        -visualize_graph $visualize_graph \
        -device 'cpu'
else
    CUDA_VISIBLE_DEVICES=$gpu_n python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -load_model_path $load_model_path \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -num_folds $num_folds \
        -report $report \
        -significance_level $significance_level \
        -visualize_graph $visualize_graph
fi