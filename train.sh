for tau in {0.1,0.3,0.5,0.7,0.9}
do
    for update_step in {20,50,100,200,300,500}
    do  
        python train.py --tau $tau --update_step $update_step
    done
done
