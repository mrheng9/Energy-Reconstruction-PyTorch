# NOvA CAF HDF5 Implement Development (Pytorch Version) 
This tutorial based on the server @tau-neutrino.ps.uci.edu  
Before this, read all papers professor send you to understand what you are doing.

 ![image](https://user-images.githubusercontent.com/80438168/169384112-ba0c39ed-f50a-4a03-bf0f-301b3690cc56.png)
https://news.fnal.gov/2014/10/fermilabs-500-mile-neutrino-experiment-up-and-running/

## Connect to Server
1, To connect the server, you must be on the UCI network. You can access the network with a VPN if you aren't on campus, info available here: https://www.oit.uci.edu/services/security/vpn/.

2, Contact maintainer of the server to get an account on the server such as you@tau-neutrino.ps.uci.edu, and an initial password.  

3, Connect to the server with your initial password:  ```$ssh you@tau-neutrino.ps.uci.edu```.

4, Change your initial password and follow prompts:  ```$passwd```.

*Note: Be sure your password is safe and correct. Your account will be locked when you input wrong password twice.  
You are not entitled to run at root or sudo.

## Deploy the repo to the server under your directory
1, You need to be able to access github from the server. You can generate a personal token in GitHub Settings -> Developer settings -> Personal access tokens -> Tokens (classic) -> Generate new token -> Generate new token (classic). Remember to copy and save the generated token because it will only be displayed once.

2, Your directory path on the server is /home/you/, init your remote repo    
```$ git clone url_to_repo```
where url_to_repo is ```https://github.com/zhongyiwu/nova.git```, or that replacing "zhongyiwu" with your own username if you forked your own version (don't worry about that point if you're just starting though). You will be asked for your GitHub username and password. Your username should be your normal username, but the password should be the personal token you just generated. Now, this repo should appear as the directory "nova". 
All tensorflow tutorial files are packed in the "tensorflow" directory in case you don't have them. If you have gone through the tensorflow version in the tau server, you can only keep the "pytorch_version".

3, You can follow the same procedure to clone this repo to your local computer.

## Setup Environment
### Add the following to your profile or `~/.bashrc` (skip if you have gone through the tensorflow tutorial):
```
# NOvA repository
NOVA=/home/you/nova/
export PYTHONPATH=$PYTHONPATH:$NOVA
```

### Install torch module
Since the tau server doesn't have the torch module installed, you will need to install it manually:
```
$ pip install torch==2.7.0 torchvision torchaudio
```
Check if the module is properly installed:
```
$ pip show torch
```
Make sure your numpy version is compatible with torch (<2.0):
```
$ pip show numpy
```

### Prepare directories for models and .pkl files
```
$ cd /home/you/nova/pytorch_version
$ mkdir models
$ mkdir pkl_file
```

### Artifacts
```
-nova
    \
    -pytorch_version
        \
        -Googlenet_Regression.py
        -Mobilenet_Regression.py
        -Preprocessed_Pytorch.py
        -test2.py
        -train2.py
        -models
        -pkl_file
```

### If you need to copy files from your local computer to the remote server:
```
$ scp /local/path/ yourname@tau-neutrino.ps.uci.edu:/home/you/nova/pytorch_version
```

## Training
### Run 
Our data directory on the server: `/mnt/ironwolf_20t/users/yuechen/data/`  

First, prepare your directories and configure the training script:
```bash
$mkdir -p yourbase/models/debug  # That's where you save the trained model and weights
$cd <yourbase>
```

Open train2.py with nano: `$ nano train2.py` (or other editors you are familiar with), and change the saving path of model to your own path `/home/you/nova/pytorch_version/models`. You can also make your own directory and add the saving path of the log file in the python file.

#### Different ways to run the model 
(choose the one you prefer)

**First: nohup**
```bash
$nohup python train2.py --path <path to data folder> --model <model_type> --name <pick a name> &
```
For example :
```bash
$nohup python train2.py --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10_train --model googlenet --name train &
```
(Change the path to yourbase/data/after_process_Jan10_train if you have your own dataset)   
Check running display information in nohup.out

If needed, add the name to the log file:
```bash
$nohup python train2.py --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10_train --model googlenet --name train > train.log 2>&1 &
```
This is a great way to keep your logs tidy. It ensures that the output from each training process gets its own file, which is perfect for when you're running different jobs on different GPUs at the same time and want to keep things separate.  

Check running display information in train.log

***

**Second: Tmux (Alternative way of nohup): Connect to Server without piping off**
```bash
$tmux new -s <session_name>  # Create new session with custom name
```
For example (This allows you to run long processes and detach/reattach as needed):
```bash
$tmux new -s ml_training  # Create a new session named "ml_training"
```
Once inside the tmux session, you can run your training command:
```bash
$python train2.py --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10_train --model googlenet --name train | tee train.log
```

Managing tmux sessions:
```bash
# Detach from current session (keeps processes running)
$tmux detach
# Or use keyboard shortcut: Ctrl+b, then d

# List all active sessions
$tmux ls

# Reattach to a specific session
$tmux attach -t ml_training

# Switch between sessions (when inside tmux)
$tmux switch -t ml_training

# Kill a session when no longer needed
$tmux kill-session -t ml_training
```

If needed, you can also redirect output within tmux:
```bash
$python train2.py --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10_train --model googlenet --name train > train.log 2>&1
```
***

**Third: tee (Real-time monitoring with log)**
```bash
$python train2.py --path <path to data> --model <model_type> --name <model_name> | tee log_file_name.log
```
For example:
```bash
$python train2.py --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10_train --model googlenet --name train | tee train.log
```

### More functions
#### for mode (nue/electron):
Change ```--mode``` nue to ```--mode``` electron:
```
$nohup python3.6 train.py --mode electron --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10_train --name electron_train_output &
```

#### for model types:
Model options are `resnet` ,`--googlenet` and `--mobilenet`. You can choose arbitrary names for your model and log file.
```bash
$nohup python train2.py --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10_train --model mobilenet --name mobile_train_output &
```

#### for weighted training:
Add the `--weighted` flag to run weighted training:
```bash
$nohup python train2.py --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10_train  --weighted --name train_weighted &
```
For example, to run weighted training with tmux:


You can check the run status in your log file, and the output model file will be saved in the models directory.

## Testing
After the training, you will see .pt files in your "models" directory. 

To test the models:
```bash
$mkdir -p yourbase/predictions/  # Create directory for test results
```

Open test2.py and change the saving path for the output .pkl file to your own path.

```bash
$nohup python test2.py --modelpath <your training results> --model <model_type> --path <testing file folder> --name <pick_a_name> &
```
For example:
```bash
$nohup python test2.py --modelpath /home/you/nova/pytorch_version/models/train.pt --model googlenet --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10_test --name test &
```

If needed, add the name to the log file:
```bash
$nohup python test2.py --modelpath /home/you/nova/pytorch_version/models/train.pt --model googlenet --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10_test --name test > test.log 2>&1 &
```

**Note:** 
- You need to use the same model_type for testing as for training
- You can use either the same or different set of files for testing
- There is no `--weighted` option in testing phase
- The output .pkl files will be saved to "pkl_file" directory
- Check running display information in nohup.out or your specified log file
- Check testing result in yourbase/predictions/{}_{}.pkl

## Plotting
### Run Jupyter notebook
Locally, run ```$ ssh -L 8888:localhost:8889 you@tau-neutrino.ps.uci.edu```.

Remotely, run ```$ jupyter notebook --no-browser --port=8889```.

Copy the url to your browser. In your browser, change http://localhost:8889/ to http://localhost:8888/.

### Analysis test result in Jupyter
It’s necessary to understand how to use matplot library and scipy statistic tool  
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html?highlight=norm

#### Jupyter essential shortcuts:  
```Ctrl + Enter - executes the current cell  
Ctrl+S - saves the notebook and checkpoint  
Enter switch to edit mode  
h - it shows all keyboard shortcuts  
a -	above new cell  
b -	below new cell  
```

### The notebook `plots_nue_energy.ipynb` helps with making plots.
If you want to make a new plot, simply use `load(fname)` with the file-name outputted at the end of running `test.py` (above). Then copy the code from the existing plots to make a new plot. The loaded object has keys `y` for the true energies, `yhat` for predicted energies, and `resolution` for `yhat/y`.
