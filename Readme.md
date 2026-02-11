# 🦾 Learning-Based Task Planning for Dual-Arm Manipulation

This project proposes a learning-driven task planner for dual-arm robotic manipulation, designed to overcome the combinatorial explosion that affects conventional task and motion planning (TAMP) approaches.

## 🚀 Description

Dual-arm robots are increasingly used to manipulate everyday household objects. To act effectively, these robots rely on task and motion planning, which combines:

- Discrete task planning — deciding which actions to take
- Continuous motion planning — computing how to move safely and efficiently

Discrete task planning struggles as the number of possible object and robot states grows. This project introduces an efficient alternative that learns task plans directly from human demonstrations.

The human demonstrations are taken from the [BiCap dataset](https://davidevdual.github.io/BiCap/).

## 🧠 Key Contributions

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

## 🧩 Project Highlights

- ✔️ 4k+ human task plans annotated using a structured grammar
- ✔️ Learned task planner outperforms a classical AI planner
- ✔️ End-to-end system deployed on a real robot
- ✔️ Reduction of combinatorial explosion via demonstration learning

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

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