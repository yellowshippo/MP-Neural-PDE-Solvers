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
	$(PYTHON) -m pip install torch==1.11.0 torchvision==0.2.2 -f https://download.pytorch.org/whl/torch_stable.html \
  && $(PYTHON) -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html \
  && $(PYTHON) -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html \
  && $(PYTHON) -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html \
  && $(PYTHON) -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html \
  && $(PYTHON) -m pip install torch-geometric

## Install PyTorch geometric with GPU
install_geo_gpu: poetry
	$(PYTHON) -m pip install torch==1.9.0+$(CUDA) torchvision==0.7.0+$(CUDA) -f https://download.pytorch.org/whl/torch_stable.html \
  && $(PYTHON) -m pip install llvmlite==0.35.0 \
  && $(PYTHON) -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+$(CUDA).html \
  && $(PYTHON) -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+$(CUDA).html \
  && $(PYTHON) -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+$(CUDA).html \
  && $(PYTHON) -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+$(CUDA).html \
  && $(PYTHON) -m pip install torch-geometric

poetry:
	$(PIP) install pip --upgrade
	poetry config virtualenvs.in-project true

run: install
	$(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=6

we3: install
	$(PYTHON) experiments/train.py --device=cpu --experiment=WE3 \
		--base_resolution=250,40 --neighbors=6 --time_window=25 --log=True
