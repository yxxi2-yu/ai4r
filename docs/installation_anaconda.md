# Autonomous Driving Environment Setup using Anaconda
This small guide summarises the steps involved for those using the Anaconda Python Distribution to setup your RL Training environment. This should work whether or not you already have python installed. The steps mentioned here should be self-contained.

> *This setup is using the Anaconda Python Distribution however, if you prefer the python/virtualenv setup then you can refer to Paul's installation notes [here]( https://gitlab.unimelb.edu.au/ai4r/ai4r-gym/-/blob/main/docs/installation.md).*

### Table of Contents

| OS       | Link                                              |
|----------|---------------------------------------------------|
| Windows  | [Environment Setup for Windows Users](#environment-setup-using-anaconda-for-windows-users) |
| macOS    | [Environment Setup for macOS Users](#environment-setup-using-anaconda-for-macos-users)     |
| Linux    | [Environment Setup for Linux Users](#environment-setup-using-anaconda-for-linux-users)     |


## Environment Setup using Anaconda (for Windows Users)

### Install and setup environment
*(These steps only need to be performed once to setup the environment and dependencies)*
- Install and setup Anaconda from the following link: https://www.anaconda.com/download (if you already have it installed, skip to the next step)
- Once installed, open 'Anaconda Prompt' from the start menu or 'CMD prompt' from the Anaconda Navigator
- **Create a virtual environment named 'gym'** with the following command:  
  `conda create --name gym python=3.10`
- Activate the 'gym' environment with the following command:  
  `conda activate gym`  
  if the environment is successfully activated, a `(gym)` sign should appear to the left side of the command line.
- Run the following command to install dependencies:
  ```
  conda install -c conda-forge gymnasium stable-baselines3 tqdm scipy tensorboard notebook
  ```
- **Cloning the `ai4r-gym` environment from gitlab:** In order to clone the environment, run the following command:  
  `git clone https://gitlab.unimelb.edu.au/ai4r/ai4r-gym.git`  
  This should download/clone the environment inside the current directory (`C:\Users\<USERNAME>\ai4r-gym`)
- Download the notebook `autonomous_driving_gym_v2024_08_21.ipynb` and place it inside the `ai4r-gym` directory

You should now have all the necessary packages installed to train your own RL models. You have a few different options now. 

### Option 1: Use Jupyter Notebooks (Similar to Google Colab)
- Open the Anaconda Prompt
- Activate the envrionment: `conda acivate gym`
- Change directory to the 'ai4r-gym': `cd ai4r-gym`
- To start the jupyter notebook server: `jupyter notebook`  
  *Remember not to close this window or else your notebook won't work*
- A browser window should pop-up showing the Jupyter Interface. Select the `autonomous_driving_gym_v2024_08_21.ipynb` file from there and open it. The rest should be similar to google colab.
- To close the jupyter notebook, close the tabs and press `Ctrl + C` in the terminal
- To deactivate the environment: `conda deactivate`

> ### ⚠️ **Important Note: If you are running Local Notebooks**
> 
> The original notebook was developed keeping Google Colab in mind, where we need to do 3 steps at the beginning of each run - 
> 1. Install gym and stable_baselines
> 2. Clone ai4r-gym repository
> 3. Change Directories
>
> If you are running notebooks locally, you will NOT need to run the first 3 cells of the notebook, and running them might just cause further complications/errors. So it is best to either comment out these 3 lines of code - 
>
>```
>#!pip install -q gymnasium stable_baselines3
>#~git clone https://gitlab.unimelb.edu.au/ai4r/ai4r-gym.git
>#%cd ai4r-gym
>```
>
>or just delete these 3 cells. 

### Option 2: Using VSCode (Visual Studio Code)
- Download and Install VSCode from the following link: https://code.visualstudio.com/  
  (Ignore if you already have it installed)
- Install the Python, Jupyter Extensions (Press `Ctrl + Shift + X` to open up the extensions window)

#### 2.1: Python Scripts
- You can open the `ai4r-gym` directory with visual studio code
- Press `Ctrl + Shift + P` to bring up the command palette and type in: "`Python: Select Interpreter`" - click on the option and a list of environments shouls show up. Select the `gym` environment. 
- Now you can run the `train.py` or the `eval.py` files from the terminal (For terminal, press `Ctrl + Shift + `) with commands such as - `python train.py`

#### 2.2: Notebooks
- Open the `ai4r-gym` directory with visual studio code, make sure your notebook is inside this directory
- Make sure that you have the Python and Jupyter Extensions installed
- Open the notebook, in the top-right, press the `Select Kernels` button > Python Environments and pick your `gym` environment
- The notebook should now run within your environment. 

### Little things to keep in mind

#### Activating / Deactivating Environment
By default, whenever you launch the command prompt, it launches into the `(base)` environment. Since we have installed our dependencies into the `gym` environment, each time you launch the prompt it is recommended that you activate the `gym` environment using the following command: `conda activate <ENV_NAME>` in our case, `conda activate gym`
(you might not need this step if you are using VSCode)

#### Directory Navigation
By default, the Anaconda Prompt should launch into a directory like this: `C:\Users\<USERNAME>\`. If you want to choose a different directory to keep the repository/models in you might need to change the directory. The command for changing directories is: `cd <DIRECTORY_NAME>` to go up one directory, you can type `cd ..`


## Environment Setup using Anaconda (for macOS Users)

### Install and setup environment
*(These steps only need to be performed once to setup the environment and dependencies)*
- Install and setup Anaconda from the following link: https://www.anaconda.com/download (if you already have it installed, skip to the next step)
- Once installed, open 'Terminal' from your Applications
- **Create a virtual environment named 'gym'** with the following command:  
  `conda create --name gym python=3.10`
- Activate the 'gym' environment with the following command:  
  `conda activate gym`  
  if the environment is successfully activated, a `(gym)` sign should appear to the left side of the command line.
- Run the following command to install dependencies:
  ```
  conda install -c conda-forge gymnasium stable-baselines3 tqdm scipy tensorboard notebook
  ```
- **Cloning the `ai4r-gym` environment from gitlab:** In order to clone the environment, run the following command:  
  `git clone https://gitlab.unimelb.edu.au/ai4r/ai4r-gym.git`  
  This should download/clone the environment inside the current directory (`~/ai4r-gym`)
- Download the notebook `autonomous_driving_gym_v2024_08_21.ipynb` and place it inside the `ai4r-gym` directory

You should now have all the necessary packages installed to train your own RL models. You have a few different options now. 

### Option 1: Use Jupyter Notebooks (Similar to Google Colab)
- Open the Terminal
- Activate the environment: `conda activate gym`
- Change directory to the 'ai4r-gym': `cd ai4r-gym`
- To start the Jupyter notebook server: `jupyter notebook`  
  *Remember not to close this window or else your notebook won't work*
- A browser window should pop-up showing the Jupyter Interface. Select the `autonomous_driving_gym_v2024_08_21.ipynb` file from there and open it. The rest should be similar to google colab.
- To close the Jupyter notebook, close the tabs and press `Ctrl + C` in the terminal
- To deactivate the environment: `conda deactivate`

> ### ⚠️ **Important Note: If you are running Local Notebooks**
> 
> The original notebook was developed keeping Google Colab in mind, where we need to do 3 steps at the beginning of each run - 
> 1. Install gym and stable_baselines
> 2. Clone ai4r-gym repository
> 3. Change Directories
>
> If you are running notebooks locally, you will NOT need to run the first 3 cells of the notebook, and running them might just cause further complications/errors. So it is best to either comment out these 3 lines of code - 
>
>```
>#!pip install -q gymnasium stable_baselines3
>#~git clone https://gitlab.unimelb.edu.au/ai4r/ai4r-gym.git
>#%cd ai4r-gym
>```
>
>or just delete these 3 cells. 

### Option 2: Using VSCode (Visual Studio Code)
- Download and Install VSCode from the following link: https://code.visualstudio.com/  
  (Ignore if you already have it installed)
- Install the Python, Jupyter Extensions (Open up the Extensions window by pressing `Cmd + Shift + X`)

#### 2.1: Python Scripts
- You can open the `ai4r-gym` directory with Visual Studio Code
- Press `Cmd + Shift + P` to bring up the command palette and type in: "`Python: Select Interpreter`" - click on the option and a list of environments should show up. Select the `gym` environment. 
- Now you can run the `train.py` or the `eval.py` files from the terminal (For terminal, press `Cmd + Shift + ``) with commands such as - `python train.py`

#### 2.2: Notebooks
- Open the `ai4r-gym` directory with Visual Studio Code, make sure your notebook is inside this directory
- Make sure that you have the Python and Jupyter Extensions installed
- Open the notebook, in the top-right, press the `Select Kernels` button > Python Environments and pick your `gym` environment
- The notebook should now run within your environment. 

### Little things to keep in mind

#### Activating / Deactivating Environment
By default, whenever you launch the Terminal, it launches into the `(base)` environment. Since we have installed our dependencies into the `gym` environment, each time you launch the Terminal it is recommended that you activate the `gym` environment using the following command: `conda activate <ENV_NAME>` in our case, `conda activate gym`
(you might not need this step if you are using VSCode)

#### Directory Navigation
By default, the Terminal should launch into your home directory. If you want to choose a different directory to keep the repository/models in you might need to change the directory. The command for changing directories is: `cd <DIRECTORY_NAME>` to go up one directory, you can type `cd ..`

Here's a revised version of your guide for setting up an RL Training environment using Anaconda, specifically tailored for Linux users:

## Environment Setup using Anaconda (for Linux Users)

### Install and Setup Environment
*Follow these steps once to set up the environment and dependencies:*

- **Install Anaconda**: Download Anaconda for Linux from [https://www.anaconda.com/download](https://www.anaconda.com/download). Follow the installation instructions on the website. If Anaconda is already installed, skip to the next step.
- **Open Terminal**: Use your preferred method to open a terminal window.
- **Create a Virtual Environment**: Create an environment named 'gym' with the command:
  ```
  conda create --name gym python=3.10
  ```
- **Activate the 'gym' Environment**: Use the command:
  ```
  conda activate gym
  ```
  You should see `(gym)` appear at the beginning of your command prompt if the activation is successful.
- **Install Dependencies**: Run the following command to install necessary packages:
  ```
  conda install -c conda-forge gymnasium stable-baselines3 tqdm scipy tensorboard notebook
  ```
- **Clone the `ai4r-gym` Environment**: To clone the environment, execute:
  ```
  git clone https://gitlab.unimelb.edu.au/ai4r/ai4r-gym.git
  ```
  This clones the environment into the current directory, typically `~/ai4r-gym`.

- **Download the Notebook**: Download `autonomous_driving_gym_v2024_08_21.ipynb` and place it inside the `ai4r-gym` directory.

Now your environment is ready to train RL models.

### Option 1: Use Jupyter Notebooks (Similar to Google Colab)
- **Open Terminal** and activate the environment:
  ```
  conda activate gym
  ```
- **Navigate to the `ai4r-gym` directory**:
  ```
  cd ~/ai4r-gym
  ```
- **Start the Jupyter Notebook server**:
  ```
  jupyter notebook
  ```
  A browser window should open with the Jupyter interface. Select and open the `autonomous_driving_gym_v2024_08_21.ipynb` file.
- **Close the Jupyter Notebook**: Close the browser tabs and terminate the server in the terminal with `Ctrl + C`.
- **Deactivate the environment**:
  ```
  conda deactivate
  ```

> ### ⚠️ **Important Note: If you are running Local Notebooks**
> 
> The original notebook was designed for Google Colab. Do not execute the first 3 cells when running locally to avoid complications. Consider commenting out or deleting these cells:
> 
> ```
> #!pip install -q gymnasium stable_baselines3
> #!git clone https://gitlab.unimelb.edu.au/ai4r/ai4r-gym.git
> #%cd ai4r-gym
> ```

### Option 2: Using VSCode (Visual Studio Code)
- **Install VSCode**: Download and install from [https://code.visualstudio.com/](https://code.visualstudio.com/). Skip if already installed.
- **Install Extensions**: Install the Python and Jupyter extensions by accessing the extensions view (`Ctrl + Shift + X`).

#### 2.1: Python Scripts
- **Open the `ai4r-gym` directory in VSCode**.
- **Select the Python Interpreter**: Use `Ctrl + Shift + P`, type "`Python: Select Interpreter`", and choose the `gym` environment.
- **Run Python scripts** such as `train.py` from the integrated terminal (`Ctrl + Shift + ``).

#### 2.2: Notebooks
- **Open and configure the notebook**: Ensure the `ai4r-gym` directory is open in VSCode. Open the notebook and use the `Select Kernel` button to choose the `gym` environment.

### Little Things to Keep in Mind

#### Activating / Deactivating the Environment
- **Switch to the `gym` Environment**: Always activate the `gym` environment in the terminal with:
  ```
  conda activate gym
  ```

#### Directory Navigation
- **Change Directories**: If your files are in a different directory, navigate using:
  ```
  cd <DIRECTORY_PATH>
  ```