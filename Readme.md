# Learning-Based Task Planning for Dual-Arm Manipulation

This project proposes a learning-driven task planner for dual-arm robotic manipulation, designed to overcome the combinatorial explosion that affects conventional task and motion planning (TAMP) approaches.

## Description

Dual-arm robots are increasingly used to manipulate everyday household objects. To act effectively, these robots rely on task and motion planning, which combines:

- Discrete task planning — deciding which actions to take
- Continuous motion planning — computing how to move safely and efficiently

Discrete task planning struggles as the number of possible object and robot states grows. This project introduces an efficient alternative that learns task plans directly from human demonstrations.

The human demonstrations are taken from the [BiCap dataset](https://davidevdual.github.io/BiCap/).

## Key Contributions

**1. Human-Demonstrated Task Plan Dataset**

15 participants performed three Activities of Daily Living (ADLs).
An RGB camera recorded hand-object interactions.
A human expert annotated 4,026 task plans using a Bio-Inspired Action Context-Free Grammar (BACFG).

**2. LSTM-Based Task Planner**

A Long Short-Term Memory (LSTM) neural network was trained on the annotated demonstrations.
The model learns to infer symbolic task plans for both seen and unseen manipulation goals.
This replaces slow symbolic search with fast sequence prediction.

**3. Performance Evaluation**

Four experiments compared the LSTM-based planner with Fast Downward, a classic symbolic task planner.
Results show that the learnt planner significantly reduces task planning time.

**4. Full Robot Integration**

The learnt task planner was integrated with an RRT (Rapidly Exploring Random Tree) motion planner.
A custom task execution framework couples high-level plans and robot motions.
The full pipeline was deployed on a dual-arm robot prototype.

## Project Highlights

- ✔️ 4k+ human task plans annotated using a structured grammar
- ✔️ Learned task planner outperforms a classical AI planner
- ✔️ End-to-end system deployed on a real robot
- ✔️ Reduction of combinatorial explosion via demonstration learning

## Getting Started

### Hardware and Software Requirements
* The learning-based planner was trained, tested, and evaluated against Fast Downward on a Windows 11 machine with an Intel i7-10875H CPU @2.30GHz and an NVIDIA GeForce RTX 2060 GPU.
* Microsoft Visual Studio Community 2022 Version 17.14.20

### Dependencies

* * Python 3.10
  * Numpy 2.2.6
  * OpenCV-Python 4.11.0.86
  * PyTorch 2.6.0+cu124
* Cuda compilation tools, release 12.4, V12.4.99
* Rest of the dependencies are listed in requirements.txt
* Fast Downward planning system 20.06
  * Download from: http://www.fast-downward.org/ObtainingAndRunningFastDownward
  * Follow the instructions to build the planner

### Installing

* Install Visual Studio Community 2022 or the latest version from https://visualstudio.microsoft.com/downloads/
* Download and build the Fast Downward planning system 20.06 planner as per the instructions on their website
* 
* Install all the dependencies listed above
* Open the project solution file LfD_Planner.sln in Visual Studio. All the files are organised in the solution explorer

### Models


### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)