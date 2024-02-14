# ./run-cifar100.sh model_name learning_rate batch_size
# ./run-cifar100.sh resnet18_bn 0.025 32

model=$1
lr=$2
bs=$3

if [ $model == "knresnet18" ] || [ $model == "knresnet34" ] || [ $model == "knresnet50" ]
then
        python3 ../simulate.py --dataset cifar100 --low-resolution --num-classes 100 --model $model \
                    --optimizer sgd --momentum 0.9 --learning-rate $lr --batch-size $bs --epochs 150 \
                    --random-hflip --random-crop 32x32-4x4 \
                    --loss cross_entropy  --weight-decay 0.0005 \
                    --lr-scheduler cosine_annealing --decay-epochs 150 --decay-multiplier 0.01 \
                    --checkpoint-freq 1 --run 1
else
        python3 ../simulate.py --dataset cifar100 --low-resolution --num-classes 100 --model $model \
                    --optimizer sgd --momentum 0.9 --learning-rate $lr --batch-size $bs --epochs 150 \
                    --random-hflip --random-crop 32x32-4x4 \
                    --norm-mean 0.5071,0.4865,0.4409  --norm-std 0.2673,0.2564,0.2762 \
                    --loss cross_entropy  --weight-decay 0.0005 \
                    --lr-scheduler cosine_annealing --decay-epochs 150 --decay-multiplier 0.01 \
                    --checkpoint-freq 1 --run 1
fi

python3 simulate.py --dataset cifar100 --low-resolution --num-classes 100 --model $model \
                    --optimizer sgd --momentum 0.9 --learning-rate $lr --batch-size $bs --epochs 150 \
                    --random-hflip --random-crop 32x32-4x4 \
                    --norm-mean 0.5071,0.4865,0.4409  --norm-std 0.2673,0.2564,0.2762 \
                    --loss cross_entropy  --weight-decay 0.0005 \
                    --lr-scheduler cosine_annealing --decay-epochs 150 --decay-multiplier 0.01 \
                    --checkpoint-freq 1 --run 1