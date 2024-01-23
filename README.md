## Lab G7 - Natural selection (MAIN LAB TO BE EVALUATED)
Simulator developed for the course CASL - Computer-Aided Simulations Lab

### Proposed activity
Consider a simulator for natural selection with the following simplified simulation model:

* All the individuals belong to S different species
* Let s be the index of each species, s=1...S
* The initial population of s is equal to P(s)
* The reproduction rate for each individual is $\lambda$
* The theoretical lifetime $\text{LF}(k)$ of individual $k$ whose parent is $d(k)$ is distributed according to the following distribution:

$ \text{LF}(k) \sim U[\text{LF}(d(k)),\text{LF}(d(k))\cdot(1+\alpha)] $ with probability $\text{prob}_\text{improve} \newline$
$ \text{LF}(k) \sim U[0,\text{LF}(d(k))] $ with probability $1-\text{prob}_\text{improve} \newline$
where $\text{prob}_\text{improve}$  is the probability of improvement for a generation and $\alpha$ is the improvement factor (>=0)

* The individuals move randomly in a given region and when individuals of different species meet, they may fight and may not survive. In such a case, the actual lifetime of a individual may be lower than its theoretical lifetime. A dead individual cannot reproduce.

Answer to the following questions:
1. Describe some interesting questions to address based on the above simulator.
2. List the corresponding input metrics.
3. List the corresponding output metrics.
4. Describe in details the **mobility model with finite speed**
5. Describe in details the **fight model** and the **survivabilty model**
6. Develop the simulator
7. Define all the events you used and their attribute.
8. Define some interesting scenario in terms of input parameters.
9. Show and comment some numerical results addressing the above questions.
