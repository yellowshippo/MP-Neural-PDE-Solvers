CUDA = cu111  # CUDA version 11.1
PYTHON ?= poetry run python3
PIP ?= pip3

SAVE_DIRECTORY = none
HIDDEN_FEATURES ?= 128  # 32 64 128
NEIGHBORS ?= 4  # 4 8 16
DATA_TYPE ?= ref


install: poetry install_geo_gpu
	$(PYTHON) -m pip install h5py
	$(PYTHON) -m pip install -e .

install_cpu: poetry install_geo_gpu_cpu
	$(PYTHON) -m pip install h5py
	$(PYTHON) -m pip install -e .

## Install PyTorch geometric with CPU
install_geo_cpu:
	$(PYTHON) -m pip install torch==1.9.1 torchvision==0.10.1 --extra-index-url https://download.pytorch.org/whl/cpu \
		&& $(PYTHON) -m pip install -U torch-scatter torch-sparse==0.6.12 torch-cluster torch-spline-conv torch-geometric \
		-f https://data.pyg.org/whl/torch-1.9.1+cpu.html

## Install PyTorch geometric with GPU
install_geo_gpu:
	-$(PYTHON) -m pip uninstall -y torch-cluster torch-scatter torch-sparse torch-spline-conv torch-geometric
	-$(PYTHON) -m pip uninstall -y torch-cluster torch-scatter torch-sparse torch-spline-conv torch-geometric
	$(PYTHON) -m pip install torch==1.9.1+$(strip $(CUDA)) torchvision==0.10.1+$(strip $(CUDA)) --extra-index-url https://download.pytorch.org/whl/cu111 \
		&& $(PYTHON) -m pip install -U \
		torch-cluster==1.6.0 \
		torch-scatter==2.0.9 \
		torch-sparse==0.6.12 \
		torch-spline-conv==1.2.1 \
		torch-geometric==2.0.4 \
		-f https://data.pyg.org/whl/torch-1.9.1+$(strip $(CUDA)).html

poetry:
	poetry config virtualenvs.in-project true


# Training
train_tw8:
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=mixture \
		--time_window=8 \
		--neighbors=$(NEIGHBORS) \
		--hidden_features=$(HIDDEN_FEATURES) \
		--unrolling=0 \
		--log=True

train_tw4:
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=mixture \
		--time_window=4 \
		--neighbors=$(NEIGHBORS) \
		--hidden_features=$(HIDDEN_FEATURES) \
		--unrolling=1 \
		--log=True

train_tw2:
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=mixture \
		--time_window=2 \
		--neighbors=$(NEIGHBORS) \
		--hidden_features=$(HIDDEN_FEATURES) \
		--unrolling=2 \
		--log=True


# Evaluation
eval_tw8:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/train.py --device=cpu --experiment=mixture \
		--log=False --neighbors=$(NEIGHBORS) --time_window=8 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--save_directory=$(SAVE_DIRECTORY) \
		--pretrained_model_file=$(MODEL_PATH) \
		--data_type=$(DATA_TYPE)

eval_tw4:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/train.py --device=cpu --experiment=mixture \
		--log=False --neighbors=$(NEIGHBORS) --time_window=4 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--save_directory=$(SAVE_DIRECTORY) \
		--pretrained_model_file=$(MODEL_PATH) \
		--data_type=$(DATA_TYPE)

eval_tw2:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/train.py --device=cpu --experiment=mixture \
		--log=False --neighbors=$(NEIGHBORS) --time_window=2 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--save_directory=$(SAVE_DIRECTORY) \
		--pretrained_model_file=$(MODEL_PATH) \
		--data_type=$(DATA_TYPE)


# Test
test: install
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=mixture \
		--log=False --neighbors=1

test_param:
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=mixture \
		--time_window=$(TW) \
		--neighbors=1 \
		--hidden_features=$(HIDDEN_FEATURES) \
		--log=True

test_cpu: install_cpu
	$(PYTHON) experiments/train.py --device=cpu --experiment=mixture \
		--log=False --neighbors=1 --time_window=20

test_eval_cpu: install_cpu
	$(PYTHON) experiments/train.py --device=cpu --experiment=mixture \
		--log=False --neighbors=6 --time_window=8 --mode=predict \
		--pretrained_model_file=tests/data/pretrained/GNN_ns_mixture/model.pt
