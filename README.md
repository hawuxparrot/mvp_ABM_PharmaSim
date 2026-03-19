### Problem Overview 
The end goal is to model the pharmaceutical distribution network EU-wide to detect cross-border theft/counterfeiting of pharmaceuticals. 

### MVP Definition
The first step is to provide a small-scale (< 100 nodes) proof of concept using synthetic data to fix the general architecture of the project. We will try to make the synthetic data as compatible/realistic as possible as the format used by the EMVO and NMVOs.

We want to combine the ease-of-use and mature library ecosystem of data processing and visualization in Python, while sidestepping the performance issues that Python's interpreted nature and the GIL impose. To do so, we separate the problem into "policy" vs "mechanism", where we define the policy and initial state using python, and leave the execution to custom C++ modules.

In this MVP we implement a simplified version to showcase this approach, with reduced numbers of agents and of agent characteristics/functionality.

### Architecture
Policy -> Compiler -> Engine -> Analytics -> I/O
1. Policy (Python): initialize scenario, agent behavioral rules
2. Compiler (Python): validate policy, assign IDs, output engine-ready columns.
3. Engine (C++): SoA simulation kernel, transition rules, maintain EMVS-style transaction log
4. Analytics (C++, expose certain functionality to Python): report builders over event log -> EMVS style transaction + metric reports
5. I/O:  I/O  

User should only have to interact with Policy, Analytics and never Engine layer.

### Directory Structure
```python/policy/``` - scenario authoring models and rules
```python/compiler/``` - AoS -> SoA compile pipeline
```python/runtime/``` - Python-facing wrapper around C++ engine
```python/analytics/``` - report generation and validation
```cpp/engine/``` - SoA state, transition logic, simulator loop
```cpp/bindings/``` - pybind interface
```schemas/``` - schema versions and enum docs
```tests/``` - unit + replay + golden report tests

The python portion is managed by uv
The C++ portion is managed by CMake
Nanobind is used to expose cpp modules to python

### Usage
python-first: 
```uv run python -m pip install -e```: builds/installs extension through packaging path
```uv run pytest```
```uv run ruff check```: runs tests/lint

@Han Wu hanwuh@ethz.ch
