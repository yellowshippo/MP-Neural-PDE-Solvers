CUDA = cu111  # CUDA version 11.1
PYTHON ?= python3.9
PIP ?= python3.9 -m pip

IMAGE = registry.ritc.jp/ricos/machine_learning/siml:0.2.8


in:
	docker run -w /src -it --gpus all -v${PWD}:/src --rm $(IMAGE)

in_cpu:
	docker run -w /src -it -v${PWD}:/src --rm $(IMAGE)


install:
	$(PYTHON) -m pip install h5py
	$(PYTHON) -m pip install -e .

## Install PyTorch geometric with CPU
install_geo_cpu: poetry
	$(PYTHON) -m pip install torch==1.9.1 torchvision==0.10.1 --extra-index-url https://download.pytorch.org/whl/cpu \
		&& $(PYTHON) -m pip install -U torch-scatter torch-sparse==0.6.12 torch-cluster torch-spline-conv torch-geometric \
		-f https://data.pyg.org/whl/torch-1.9.1+cpu.html

## Install PyTorch geometric with GPU
install_geo_gpu: poetry
	$(PYTHON) -m pip install torch==1.9.1+$(strip $(CUDA)) torchvision==0.10.1+$(strip $(CUDA)) --extra-index-url https://download.pytorch.org/whl/cu111 \
		&& $(PYTHON) -m pip install -U torch-scatter torch-sparse==0.6.12 torch-cluster torch-spline-conv torch-geometric \
		-f https://data.pyg.org/whl/torch-1.9.1+$(strip $(CUDA)).html

poetry:
	$(PIP) install pip --upgrade
	poetry config virtualenvs.in-project true

test: install
	$(PYTHON) experiments/flearn.py --device=cuda:0 --experiment=fluid \
		--log=False --neighbors=1

test_cpu: install
	$(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=1

flearn: install
	$(PYTHON) experiments/flearn.py --device=cuda:0 --experiment=fluid \
		--log=False --neighbors=4

we3_cpu: install
	$(PYTHON) experiments/train.py --device=cpu --experiment=WE3 \
		--base_resolution=250,40 --neighbors=1 --time_window=25

we3: install
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=WE3 \
		--base_resolution=250,40 --neighbors=1 --time_window=25
