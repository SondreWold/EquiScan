lr = 3e-4
hidden_size = 128
layers = 2
dropout = 0.1
heads = 2

for p in 1 2 4 8 16 32 64
do
    sbatch submit.slurm --log --task "simple" --p $p --lr $lr --hidden_size $hidden_size --layers $layers --dropout $dropout --heads $heads
done

sbatch submit.slurm --log --task "length" --lr $lr --hidden_size $hidden_size --layers $layers --dropout $dropout --heads $heads