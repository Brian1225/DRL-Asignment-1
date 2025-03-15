for tau in 0.1 0.3 0.5 0.9 1;
do
    python train.py --batch_size 256 --use_wandb --n_episode 10000 --tau $tau --update_step 100
    git add .
    git commit -m "tau=$tau"
    git push
done