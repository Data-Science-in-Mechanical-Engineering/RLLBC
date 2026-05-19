![DSME-logo](./class_examples/img/DSME_logo.png)

#  Reinforcement Learning and Learning-based Control

<p style="font-size:12pt";> 
<b> Prof. Dr. Sebastian Trimpe, Dr. Friedrich Solowjow </b><br>
<b> Institute for Data Science in Mechanical Engineering(DSME) </b><br>
<a href = "mailto:rllbc@dsme.rwth-aachen.de">rllbc@dsme.rwth-aachen.de</a><br>
</p>

---
The algorithms within this library were developed in the context of the class Reinforcement Learning and Learning-based Control (RLLBC) by the Institute for Data Science in Mechanical Engineering (DSME) at RWTH Aachen University. In this class we use this library in Lectures and Exercises. Students can also use the library to expand their knowledge through self-study. All algorithms are presented via Jupyter notebooks — see the individual folder READMEs for details:

- [`class_examples/`](class_examples/README.md) — lecture and exercise companion notebooks covering MDPs, DP, MC, TD, function approximation, and learning-based control
- [`tabular_examples/`](tabular_examples/README.md) — tabular RL implementations: Policy Iteration, Value Iteration, MC Control, SARSA, Q-Learning, Dyna-Q
- [`deep_examples/`](deep_examples/README.md) — deep RL implementations: DQN, REINFORCE, A2C, TRPO, DDPG, TD3, SAC
- [`lbc_examples/`](lbc_examples/README.md) — learning-based control notebooks: LQR, Dynamics Learning, MPC, Bayesian Optimisation

You can find installation instructions below.

## Installation guide

To install the library, please follow the instructions below.

1. **Download the files**

1. **Install the latest version of Pixi** https://pixi.sh/latest/installation/
   - make sure that you install the version for the operating system that you are using

2. **Create the uv environment**
      ```setup 
      pixi install
      ```
3. **Activate the environment**
   ```setup 
   pixi shell
   ```
4. **Start up JupyterLab** from your terminal with
   ```setup 
   jupyter-lab
   ```

&rarr; Now you should be able to browse your file system for the notebooks

### Using the library on a local computer:
Once the environment has been successfully installed, the library can be easily accessed via the following steps:
1. **Navigate to the project folder** and open your terminal there. On Windows, use the powershell.
2. **Activate the environment** with 
   ```setup 
   pixi shell
   ```
3. **Start up JupyterLab** from your terminal with
   ```setup 
   jupyter-lab
   ```
You are ready to browse the library.
