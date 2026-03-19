#### Problem Overview 
The end goal is to model the pharmaceutical distribution network EU-wide to detect cross-border theft/counterfeiting of pharmaceuticals. 

The first step is to provide a small-scale (< 100 nodes) proof of concept using synthetic data to fix the general architecture of the project. We will try to make the synthetic data as compatible/realistic as possible as the format used by the EMVO and NMVOs.

We want to combine the ease-of-use and mature library ecosystem of data processing and visualization in Python, while sidestepping the performance issues that Python's interpreted nature and the GIL impose. To do so, we separate the problem into "policy" vs "mechanism", where we define the policy and initial state using python, and leave the execution to custom C++ modules.

In this MVP we implement as simplified version to showcase this approach, with reduced numbers of agents and of agent characteristics/functionality.

@Han Wu hanwuh@ethz.ch
