# INSTALLATION NOTES:

Contents of this page:

* [Too long didn't read](#tldr) summary of the commands for a command line installation.

* [Command line installation](#command-line-installation)

* [Installation on Windows using VSCode](#installation-on-windows-using-vscode)



## TL;DR

Of course you did read what is below and you are referring to this section for a reminder. For Windows users, you need to refer below for the two lines that create and then source the Python virtual environment.

```
cd /location/of/your/choosing
git clone https://gitlab.unimelb.edu.au/asclinic/ai4r-gym.git
cd ai4r-gym
mkdir venv_for_ai4rgym
python -m venv venv_for_ai4rgym/
source venv_for_ai4rgym/bin/activate
pip list
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools
pip list
pip install -r requirements.txt
pip list
pip install -e . 
pip list
```



## Command line installation

### Create python virtual environment
Follow the Python docs for venv to do this:
https://docs.python.org/3/library/venv.html

In summary, the following command creates the venv and the directory for it as necessary:
```
python -m venv /path/to/new/virtual/environment
```

For windows, this command is:
```
c:\>c:\Python35\python -m venv c:\path\to\myenv
```

Alternatively, create the directory yourself, change into the directory, and then use the command:
```
python -m venv .
```



### Activate the virtual environment 
For every new terminal that you want to use to work in this environment, you need to activate the environment using the command:

* **Ubuntu and Mac OS:** `source <venv_dir>/bin/activate`
* **Windows cmd.exe:** `C:\> <venv_dir>\Scripts\activate.bat`
* **Windows Powershell:** `PS C:\> <venv_dir>\Scripts\Activate.ps1`

Where `<venv_dir>` is the directory you used above for creating the environment.



### Deactivate when you are finished

Deactivate by typing the command
```
deactivate
```

Or simply close the terminal window.



### Clone this repository

To some convenient location on your computer
```
cd /path/to/your/git/repositories
git clone https://gitlab.unimelb.edu.au/asclinic/ai4r-gym.git
```


### Upgrade pip and setup tool

The remainder of these instructions should be performed with the `venv` activated. You know that the `venv` is activated because the start of the command line shows in brackets the name that you gave when creating the `venv`.

For interest and a double check, list what is currently installed in the `venv`:
```
pip list
```

Which should display very few packages because you only just created the `venv`, something like:
```
Package    Version
---------- -------
pip        20.2.3
setuptools 49.2.1
```

Upgrade pip as recommended by the warning:
```
python -m pip install --upgrade pip
```

Upgrade also the setuptools package:
```
python -m pip install --upgrade setuptools
```


### Install the dependencies of the ai4r-gym

The `requirements.txt` in the `ai4r-gym` git repository can be used to install the dependencies before actually install the `ai4r-gym` package itself. The command is:
```
pip install -r requirements.txt
```

Check that the dependencies were correctly installed by looking at:
```
pip list
```

This will display a much longer list now because the ai4r-gym dependencies themselves have multiple dependencies. On the list you should at least identify:
```
Package              Version
-------------------- -------
gymnasium            0.29.0
matplotlib           3.7.2
numpy                1.24.4
scipy                1.10.1
```



### (OPTIONAL) Install additional dependencies

If you wish to save a visualisation of your gymnasium simulation as a movie, then additionally install the `ffmpeg` package:

* **Ubuntu:** `sudo apt install ffmpeg`
* **MacOS Intel:** `pip install ffmpeg-python`
* **MacOS Arm:** ...google it...
* **Windows:** ...google it...

* (Optional) Pillow, only need if you wish to save visualisation as a gif:
```
pip install Pillow
```


### Install `ai4r-gym`

Finally, install the ai4r-gym package using pip
```
pip install -e <path/to/ai4r-gym>
```

Alternatively, `cd` to the location where you cloned the `ai4r-gym` repository and run the command on the current folder using:
```
pip install -e .
```

As per `pip install --help`, the `-e` option is for installing a package in editable mode:

-e, --editable <path/url>   Install a project in editable mode (i.e. setuptools "develop mode") from a local project path or a VCS url.

Confirm that the installation was successful by checking that pip list now includes the following:
```
Package              Version Editable project location
-------------------- ------- ----------------------------------------
ai4rgym              0.0.1   <path/to/ai4r-gym>
```



### Test the installation

Test that everything is working by running the example script for the pendulum gymnasium environment:
```
cd <path/to/ai4r-gym>
python examples/autonomous_driving_example.py
```


To get started with any of the environment, see the respectively named example file for that environment.



## Installation on Windows using VSCode

Running on Windows in VSCode (note that these instructions are largelt take from this [VSCode tutorial page](https://code.visualstudio.com/docs/python/python-tutorial)):

1) Install Python from [python.org](https://www.python.org/). Use the Download Python button that appears first on the page to download the latest version. For additional information about using Python on Windows, see [Using Python on Windows](https://docs.python.org/3.9/using/windows.html) at Python.org

2) Install the [VSCode Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

3) Make a folder that you use as your workspace for the ai4rgym, for example, according to [this VSCode tutorial page](https://code.visualstudio.com/docs/python/python-tutorial#_start-vs-code-in-a-workspace-folder)

4) From the VSCode Terminal clone the ai4rgym using the following command (within the folder created in Step 3):
```
git clone https://gitlab.unimelb.edu.au/ai4r/ai4r-gym.git
```

5) In VSCode, open the folder from Step 3 and then create a virtual environment in that workspace as per [this VSCode tutorial page](https://code.visualstudio.com/docs/python/python-tutorial#_create-a-virtual-environment).

6) Once the virtual environment is finished being create, select the option that automatically pops up asking whether to install dependencies from "requirements.txt". If you don't see this, then use the Teminal in VSCode to install the requirements using the command:
```
py -m pip install -r requirements.txt
```

7) From the Terminal in VSCode, install the ai4rgym into the virtual environment using the command:
```
py -m pip install -e .
```
