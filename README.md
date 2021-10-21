quantum-deep-dreaming
=====================

Authors
-------

Chong Sun <sunchong137@gmail.com>
Hannah Sim <hsim13372@gmail.com>
Abhinav Anand <abhinav.anand@mail.utoronto.ca>

Pre-requests
------------

1. Tequila
For quantum simulations.
```bash
git clone https://github.com/aspuru-guzik-group/tequila.git
cd tequila
pip install -e .
```
2. PyTorch
For neural network training. Install with ``pip``:
```bash
pip install torch
```

Installation
------------

```bash
pip install -e .
```

```python
from qdd import gen_circuit
```
