### Installation

``` Bash
# create conda environment
conda create -n marl python==3.8
conda activate marl
```

```
# install on-policy package
cd on-policy
pip install -e .
```

### Hanabi
Environment code for Hanabi is developed from the open-source environment code, but has been slightly modified to fit the algorithms used here.  
To install, execute the following:
``` Bash
pip install cffi
cd envs/hanabi
mkdir build & cd build
cmake ..
make -j
```

### Install requirements
``` Bash
pip install -r requirements
```

### Run a website to play against trained agents
``` Bash
bash web.sh
```
Then open http://localhost:8000/ to play.

