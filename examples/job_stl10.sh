#!/bin/bash
#SBATCH --partition gpu          # partition (queue)
#SBATCH --nodes 1                # number of nodes
#SBATCH --mem 170000               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 0-06:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output stl10.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error stl10.%N.%j.err  # filename for STDERR

module load cuda

echo "starting script"
#source set_up
#./scripts/cifar10_micro_final.sh
#python3 experiments.py
#python3 experiments.py searcher_comparison resnet rand 4
#cd cnn && python train_search.py --batch_size 8
python3 mnist.py --dataset stl10 --datapath /home/aprabh2s/NAS/enas/data/STL-10/
echo "complete"
