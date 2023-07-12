
python test.py \
    --bart_name facebook/bart-base \
    --model_weight ./saved_model/best_model \
    --datapath  ./Twitter10000/txt \
    --image_feature_path ./twitter10000_VinVL \
    --image_annotation_path ./Twitter10000/xml \
    --box_num 16 \
    --batch_size 32 \
    --max_len 30 \
    --normalize \
          