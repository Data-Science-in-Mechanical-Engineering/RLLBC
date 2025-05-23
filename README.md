![DSME-logo](./class_examples/img/DSME_logo.png)

#  Reinforcement Learning and Learning-based Control

<p style="font-size:12pt";> 
<b> Prof. Dr. Sebastian Trimpe, Dr. Friedrich Solowjow </b><br>
<b> Institute for Data Science in Mechanical Engineering(DSME) </b><br>
<a href = "mailto:rllbc@dsme.rwth-aachen.de">rllbc@dsme.rwth-aachen.de</a><br>
</p>

---
The algorithms within this library were developed in the context of the class Reinforcement Learning and Learning-based Control (RLLBC) by the Institute for Data Science in Mechanical Engineering (DSME) at RWTH Aachen University. In this class we use this library in Lectures and Exercises. Students can also use the library to expand their knowledge through self-study. We provide example algorithms for tabular and deep reinforcement learning in the folders "tabular_examples" and "deep_examples". All algorithms are presented via Jupyter notebooks. You can find installation instructions below. For more details on how to work with the algorithms, we refer to the descriptions in the notebooks. Furthermore, we provide examples from the lecture and exercise in the folder "class_examples".

## Installation guide

To install the library, please follow the instructions below.

1. **Download the files**

1. **Install the latest version of UV** https://docs.astral.sh/uv/#installation
   - make sure that you install the version for the operating system that you are using

2. **Create the uv environment**
      ```setup 
      uv venv -p 3.10
      ```
3. **Activate the environment** on Linux with 
   ```setup 
   source .venv/bin/activate
   ```
   or on Windows with
   ```setup
    .venv\Scripts\activate
    ```
   and install the required packages with
   ```setup
   uv pip sync ./requirements.txt
    ```
4. **Start up JupyterLab** from your terminal with
   ```setup 
   jupyter-lab
   ```

&rarr; Now you should be able to browse your file system for the notebooks

*Note*: In order to be able to render videos of the agent's performance you have to make sure to have `ffmpeg` installed.

*Warning*: pybox2d is not available for Apple Silicon devices (Mac with M1, M2, or M3 processors). When working with Apple Silicon devices, this might cause issues. For installation, remove pybox2d from the list of required packages in the `environment.yml` file.

### Using the library on a local computer:
Once the environment has been successfully installed, the library can be easily accessed via the following steps:
1. **Navigate to the project folder** and open your terminal there. On Windows, use the powershell.
2. **Activate the environment** with 
   ```setup 
   source .venv/bin/activate
   ```
   or on Windows with
   ```setup
    .venv\Scripts\activate
    ```
3. **Start up JupyterLab** from your terminal with
   ```setup 
   jupyter-lab
   ```
You are ready to browse the library.


## Dev notes

To update the requirement.txt do:
```setup
uv pip compile requirements.in \
   --universal \
   --output-file requirements.txt
```