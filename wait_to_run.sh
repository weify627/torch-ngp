id=$1
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
echo $free_mem

while [ $free_mem -lt 20000 ]; do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
    sleep 5
done
#test
test_path=~/code/torch-ngp/ngp-best-2half/results/output-785e7504b9-1024b2.ply
test_path=~/code/torch-ngp/ngp-best-2half/results/gt2.ply
#python main_sdf.py $test_path --workspace ngp-best-2half --test
python main_sdf.py ~/data/scannetpp --workspace ngp-best-sdf-w2gt-noema-40m --scene_list 7 --marching_cubes_res 1024
#python main_sdf.py ~/data/scannetpp --workspace ngp-best-negsdf-ft --scene_list 7 --use_color --marching_cubes_res 1024
#python main_sdf.py ~/data/scannetpp --workspace ngp-best-2-256-dim4-785e7504b9 --scene_list 7 --marching_cubes_res 1024
#python main_sdf.py ~/data/scannetpp --workspace tcnnencode-22-2 --scene_list 7c --marching_cubes_res 1024
#python main_sdf.py ~/data/scannetpp --workspace neus-facto-angelo-10-10-wn-785e7504b9 --scene_list 7 --marching_cubes_res 1024
#python main_sdf.py ~/data/scannetpp --workspace neus-facto-angelo-c49a8c6cff
#s=flame_salmon
#s=flame_steak
#CUDA_VISIBLE_DEVICES=$id python main4d.py fit \
    #--config=configs/dynerf_res2.yaml \
    #--data.path=/home/fangyin/data/plenoptic/$s -n=nopreset.003-nopnum-1e-4/$s
    #--data.path=/home/fangyin/data/plenoptic/$s -n=nopreset.003/$s
# Note
# ngp-best is scannetppv9-500m-epoch20-lr12_8-exp5-22
