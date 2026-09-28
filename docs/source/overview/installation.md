(gentle-intro)=
# Installation

```{note}
If you encounter any issues during installation, please open an issue on [GitHub](https://github.com/CraftJarvis/MineStudio/issues).
```

Welcome to MineStudio! Please follow the tutorial below for installation.

## 1. Install JDK 8

To ensure that the Simulator runs smoothly, JDK 8 must be installed on your system. We recommend using Conda to manage environments on Linux systems. If you don't have Conda, you can install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

```console
# Create a new conda environment (optional, but recommended)
$ conda create -n minestudio python=3.10 -y

# Activate the environment
$ conda activate minestudio

# Install OpenJDK 8
$ conda install --channel=conda-forge openjdk=8 -y
```

## 2. Install MineStudio

You can install MineStudio either from GitHub or PyPI.

**a. Install from GitHub (for the latest version):**
```console
$ pip install git+https://github.com/CraftJarvis/MineStudio.git
```

**b. Install from PyPI (for stable releases):**
```console
$ pip install minestudio
```

## 3. Install the Rendering Tool

For GPU-accelerated rendering, especially for users with NVIDIA graphics cards, we recommend installing **VirtualGL**. For other users, or for a simpler setup, **Xvfb** can be used, which supports CPU rendering (this is generally slower).

```{note}
Installing rendering tools and managing services typically requires **root** permissions. Use `sudo` before commands like `apt`, `dpkg`, and `service` where necessary.
```
There are two options:
``````{tab-set}

`````{tab-item} Xvfb
This option uses Xvfb for CPU-based rendering.

```console
$ sudo apt update
$ sudo apt install -y xvfb mesa-utils libegl1-mesa libgl1-mesa-dev libglu1-mesa-dev
```
`````

`````{tab-item} VirtualGL
This option uses VirtualGL for GPU-accelerated rendering.

```{warning}
Not all graphics cards and driver setups support VirtualGL seamlessly. If you do not have strict performance requirements or encounter issues, using the Xvfb rendering tool is often easier to install and configure.
```

**3.1. Download VirtualGL and Helper Script:**

You need to download the following:
- VirtualGL Debian package: [virtualgl_3.1_amd64.deb](https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download)
- vgl_entrypoint.sh script: [vgl_entrypoint.sh](https://github.com/CraftJarvis/MineStudio/blob/master/assets/vgl_entrypoint.sh) (You might also find this script in the `assets/` directory if you cloned the MineStudio repository).

**3.2. Install Dependencies and VirtualGL:**

```console
$ sudo apt update
$ sudo apt install -y xvfb mesa-utils libegl1-mesa libgl1-mesa-dev libglu1-mesa-dev xauth xterm
$ sudo dpkg -i virtualgl_3.1_amd64.deb
# If dpkg fails due to missing dependencies, run:
$ sudo apt -f install
```

**3.3. Configure VirtualGL:**

Shutdown your display manager (e.g., gdm, lightdm). If you are using GDM, the command is:
```console
$ sudo service gdm stop
# Or for LightDM:
# $ sudo service lightdm stop
```
Run the VirtualGL server configuration script:
```console
$ sudo /opt/VirtualGL/bin/vglserver_config
```
You will be prompted with several configuration questions.
A common configuration sequence might involve:
1.  Choosing option `1` if prompted to select a display manager configuration (e.g., for GDM/LightDM).
2.  Answering subsequent questions. The original guide suggested a sequence like: Yes, No, No, No. These questions typically relate to:
    *   Restricting access to VirtualGL to members of the `vglusers` group (Recommended: `Yes`. If you choose `Yes`, add your user to the `vglusers` group: `sudo usermod -a -G vglusers $USER`).
    *   Restricting framebuffer readback on the 3D X server to members of the `vglusers` group (Recommended: `Yes`).
    *   Disabling the XTEST extension on the 3D X server (Recommended: `No`, unless you have specific security reasons and do not require indirect rendering capabilities).
The original guide also mentioned "finally enter X", the meaning of which is unclear. Please ensure you complete all configuration steps as prompted by the script. If unsure about any option, consulting the official VirtualGL documentation or accepting defaults is advisable.

Restart your display manager:
```console
$ sudo service gdm start
# Or for LightDM:
# $ sudo service lightdm start
```

**3.4. Start VirtualGL Server Script:**
Execute the downloaded `vgl_entrypoint.sh` script. Make sure it's executable (`chmod +x vgl_entrypoint.sh`).
```console
$ bash ./vgl_entrypoint.sh
```
```{warning}
You might need to run `vgl_entrypoint.sh` each time the system restarts.
```

**3.5. Configure Environment Variables:**
Add these exports to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`) and source it or open a new terminal.
```console
export PATH="${PATH}:/opt/VirtualGL/bin"
export VGL_DISPLAY="egl" # Or try :0, or the appropriate display connected to your NVIDIA GPU
export VGL_REFRESHRATE="60" # Or your monitor's actual refresh rate
export DISPLAY=":1" # This is often used for the virtual X server display
```
Ensure these variables are set in the terminal where you run MineStudio.
`````
``````

## 4. Verify by Running the Simulator

```{hint}
The first time you run the simulator, a script may ask whether to download compiled model assets from Hugging Face. Choose 'Y' to proceed.
```

**If you are using Xvfb:**
```console
$ python -m minestudio.simulator.entry
```

**If you are using VirtualGL:**
Ensure the environment variables from step 3.5 are set.
```console
$ MINESTUDIO_GPU_RENDER=1 python -m minestudio.simulator.entry
```

If you see output similar to the following, the installation is successful:
```
Speed Test Status:
Average Time: 0.03
Average FPS: 38.46
Total Steps: 50

Speed Test Status:
Average Time: 0.02
Average FPS: 45.08
Total Steps: 100
```