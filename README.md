# NOvA CAF HDF5 Implement Development (Pytorch Version) 
This tutorial based on the server @tau-neutrino.ps.uci.edu  
Before this, read all papers professor send you to understand what you are doing.

 ![image](https://user-images.githubusercontent.com/80438168/169384112-ba0c39ed-f50a-4a03-bf0f-301b3690cc56.png)
https://news.fnal.gov/2014/10/fermilabs-500-mile-neutrino-experiment-up-and-running/

## Git clone this repo(If you have your own way to check on the codes, please skip this step)  
1, Download git from here: https://git-scm.com/downloads  
2, Navigate to the directory you would like to keep the nova(e.g. $cd /you/projects/)  
3, Clone the repository into this directory on your local computer  
```$ git clone https://github.com/zhongyiwu/nova.git```  
4, Check codes with your favorite editor or IDE(e.g. Vim, Emacs, VS Code, Pycharm etc.)  
If you use vi/vim for the first time, be sure to learn vimtutor for at least 15mins and remember quit without save command ```:q!```

## Connect to Server
1, To connect the server, you must be on the UCI network. You can access the network with a VPN if you aren't on campus, info available here: https://www.oit.uci.edu/services/security/vpn/  
2, Contact maintainer of the server to get an account on the server such as you@tau-neutrino.ps.uci.edu, and an initial password.  
3, Connect to the server with your initial password:  
```$ssh you@tau-neutrino.ps.uci.edu```  
4, Change your initial password and follow prompts:  
```$passwd```  
*Note: Be sure your password is safe and correct. Your account will be locked when you input wrong password twice.  
You are not entitled to run at root or sudo.

## Deploy the repo to the server under your directory
1, You need to be able to access github from the server. You can create an ssh key for them to identify you, by running:
```$ ssh-keygen -t ed25519 -C "your_email@example.com"```
with your own email. To get your created public key, now run
```$ cat ~/.ssh/id_ed25519.pub```
Now tell github your public key by going to github.com, account settings (in the top right), "SSH And GPG Keys", then "New SSH Key". Enter the public key you just got, and now github will know it's you.

2,  Your directory path on the server is /home/you/, init your remote repo    
```$ git clone url_to_repo```
where url_to_repo is ```https://github.com/zhongyiwu/nova.git```, or that replacing "zhongyiwu" with your own username if you forked your own version (don't worry about that point if you're just starting though). Now, this repo should appear as the directory "nova". All tensorflow tutorial files are packed in the "tensorflow" directory in case you don't have them. If you have gone through the tensorflow version, you will only need the "pytorch_version" directory.

## Setup Environment  
### Add the following to your profile or `~/.bashrc`:
```
# NOvA repository
NOVA=/home/you/nova/
export PYTHONPATH=$PYTHONPATH:$NOVA
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
#### for NuE energy: (good start)
First, open train2.py with nano: ```$ nano train2.py``` (or other editor you are familiar with), and change the saving path of model to your own path ```/home/you/nova/pytorch_version/models```. You can also make your own directory for and add the saving path of the log file in the python file.

Then run the command:
```
$ python train2.py --path /path/to/data --model model_type --name model_name | tee log_file_name.log
```

where model options are ```--googlenet``` and ```--mobilenet```. You can choose arbitrary names for your model and log file.

For example:
```
$ python train2.py --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10 --model googlenet --name train_tau_pytorch | tee train_tau_pytorch.log
```

You can check the run status in your log file, and the output model file will be saved in the models directory.

If you want to run your tasks in a tmux (detached) session:
```
$ tmux
$ bash
$ python train2.py --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10 --model googlenet --name train_tau_pytorch | tee train_tau_pytorch.log
```
## Testing
After the trainning, you will see .pt files in your "models" directory. Open test2.py and change the saving path for the output .pkl file.

To test the models:
```
$ python test2.py --modelpath /home/you/nova/models/your_pt_file.pt --path /testing/file/path --name pkl_file_name --model model_type
```

You need to use the same model_type for testing as for training, but you can use either the same or different set of files for testing. For example:
```
$ python test2.py --modelpath /home/you/nova/pytorch_version/models/train_tau_pytorch.pt --path /mnt/ironwolf_20t/users/yuechen/data/after_process_Jan10 --name pytorch_tau_test --model googlenet
```

The output .pkl files will be saved to "pkl_file" directory. 

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