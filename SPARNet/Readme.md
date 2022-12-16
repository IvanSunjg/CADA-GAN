Train:
python train.py


Test:
python test.py --gpus 1 --model sparnethd --name SPARNetHD_V4_Attn2D \
    --res_depth 10 --att_name spar --Gnorm 'in' \
    --load_size 512 --dataset_name single --dataroot test_dirs/CelebA-TestN/ \
    --pretrain_model_path ./pretrain_models/SPARNet-V16-S4-epoch20.pth \
    --save_as_dir results_CelebA-TestN/SPARNetHD_V4_Attn2D/

python test.py --gpus 2 --model sparnethd --name SPARNetHD_V4_Attn3D \
    --res_depth 10 --att_name spar3d --Gnorm 'in' \
    --load_size 512 --dataset_name single --dataroot test_dirs/CelebA-TestN/ \
    --pretrain_model_path ./pretrain_models/SPARNetHD_V4_Attn3D_net_H-epoch10.pth \
    --save_as_dir results_CelebA-TestN/SPARNetHD_V4_Attn3D/

    python test.py --gpus 1 --model sparnethd --name SPARNetHD_V4_Attn2D     --res_depth 10 --att_name spar --Gnorm 'in'     --load_size 512 --dataset_name single --dataroot test_dirs/CelebA-TestN/     --pretrain_model_path ./pretrain_models/SPARNetHD_V4_Attn2D_net_H-epoch10.pth     --save_as_dir results_CelebA-TestN/SPARNetHD_V4_Attn2D/
