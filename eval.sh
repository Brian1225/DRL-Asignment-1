for model in "checkpoints/*.pth"; 
do
    python simple_custom_taxi_env.py --model $model
done 