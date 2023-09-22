# source activate
# conda activate tracker
# ln -s /root/datasets /root/dym/OSTrack
# mv datasets data 
# python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
# 软链接预训练权重
# ln -s /root/dym/pretrained_models/ /root/dym/OSTrack
# 创建路径
# mkdir output
# mkdir output/checkpoints
# mkdir output/checkpoints/train
# mkdir output/checkpoints/train/ostrack
# 软链接权重文件
# ln -s /root/dym/weights/vitb_256_mae_mff320_64x1_got10k_ep100 /root/dym/OSTrack/output/checkpoints/train/ostrack/
# ln -s /root/dym/weights/vitb_256_mae_mff256_32x1_got10k_ep100 /root/dym/OSTrack/output/checkpoints/train/ostrack/
# ln -s /root/dym/weights/vitb_256_mae_mff320_64x1_ep300 /root/dym/OSTrack/output/checkpoints/train/ostrack/
# ln -s /root/dym/weights/vitb_256_mae_64x1_got10k_ep100 /root/dym/OSTrack/output/checkpoints/train/ostrack/