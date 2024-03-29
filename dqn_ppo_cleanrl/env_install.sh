conda create --name cleanrl python=3.10 -y
conda activate cleanrl
conda install -y scipy jupyter pandas matplotlib pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tensorboard stable-baselines3 "gymnasium[accept-rom-license, atari, other]"
