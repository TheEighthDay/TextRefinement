python -m torch.distributed.launch --nproc_per_node=2 main_clip4clip.py --do_train --num_thread_reader=8 --lr 1e-4 --batch_size=100  --batch_size_val 40 --epochs=5  --n_display=10 \
--train_csv /data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/train_list_full.csv \
--val_csv /data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/val_list_full.csv \
--test_csv /data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/test_list_full.csv \
--data_path /data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_data.json \
--features_path /data/tiankaibin/Text2VideoDataset/MSRVTT/data/MSRVTT/videos/frames \
--output_dir /data/tiankaibin/xcliplog/ckpts_dsw/clip4clip_msrvttfull_vit32 \
--max_words 32 --max_frames 12 --datatype msrvtt --expand_msrvtt_sentences --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0  --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32

#python main_clip4clip.py --do_train --num_thread_reader=8 --lr 1e-4 --batch_size=60  --batch_size_val 40 --epochs=5  --n_display=10 --train_csv /data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/train_list_full.csv --val_csv /data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/val_list_full.csv --test_csv /data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/test_list_full.csv --data_path /data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_data.json --features_path /data/tiankaibin/Text2VideoDataset/MSRVTT/data/MSRVTT/videos/frames --output_dir /data/tiankaibin/xcliplog/ckpts_dsw/clip4clip_msrvttfull_vit32 --max_words 32 --max_frames 12 --datatype msrvtt --expand_msrvtt_sentences --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0  --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32