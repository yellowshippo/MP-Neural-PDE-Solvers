CUDA = cu111  # CUDA version 11.1
PYTHON ?= poetry run python3
PIP ?= pip3

SAVE_DIRECTORY = none
HIDDEN_FEATURES ?= 128  # 32 64 128


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
	$(PYTHON) -m pip install torch==1.9.1+$(strip $(CUDA)) torchvision==0.10.1+$(strip $(CUDA)) --extra-index-url https://download.pytorch.org/whl/cu111 \
		&& $(PYTHON) -m pip install -U torch-scatter torch-sparse==0.6.12 torch-cluster torch-spline-conv torch-geometric \
		-f https://data.pyg.org/whl/torch-1.9.1+$(strip $(CUDA)).html

poetry:
	$(PIP) install pip --upgrade
	poetry config virtualenvs.in-project true


# Training
train_tw20:
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=fluid \
		--time_window=20 \
		--neighbors=11 \
		--hidden_features=$(HIDDEN_FEATURES) \
		--log=True

train_tw10:
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=fluid \
		--time_window=10 \
		--neighbors=6 \
		--hidden_features=$(HIDDEN_FEATURES) \
		--log=True

train_tw4:
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=fluid \
		--time_window=4 \
		--neighbors=2 \
		--hidden_features=$(HIDDEN_FEATURES) \
		--log=True

train_tw2:
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=fluid \
		--time_window=2 \
		--neighbors=1 \
		--hidden_features=$(HIDDEN_FEATURES) \
		--log=True


# Evaluation
eval_tw20:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/train.py --device=cpu --experiment=fluid \
		--log=False --neighbors=11 --time_window=20 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--save_directory=$(SAVE_DIRECTORY) \
		--pretrained_model_file=$(MODEL_PATH)

eval_tw10:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/train.py --device=cpu --experiment=fluid \
		--log=False --neighbors=6 --time_window=10 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--save_directory=$(SAVE_DIRECTORY) \
		--pretrained_model_file=$(MODEL_PATH)

eval_tw4:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/train.py --device=cpu --experiment=fluid \
		--log=False --neighbors=2 --time_window=4 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--save_directory=$(SAVE_DIRECTORY) \
		--pretrained_model_file=$(MODEL_PATH)

eval_tw2:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/train.py --device=cpu --experiment=fluid \
		--log=False --neighbors=1 --time_window=2 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--save_directory=$(SAVE_DIRECTORY) \
		--pretrained_model_file=$(MODEL_PATH)

transformed_eval_tw20:
	$(PYTHON) experiments/train.py --device=cpu --experiment=fluid \
		--log=False --neighbors=11 --time_window=20 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--pretrained_model_file=$(MODEL_PATH) \
		--save_directory=$(SAVE_DIRECTORY) \
		--transformed=true

transformed_eval_tw10:
	$(PYTHON) experiments/train.py --device=cpu --experiment=fluid \
		--log=False --neighbors=6 --time_window=10 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--pretrained_model_file=$(MODEL_PATH) \
		--save_directory=$(SAVE_DIRECTORY) \
		--transformed=true

transformed_eval_tw4:
	$(PYTHON) experiments/train.py --device=cpu --experiment=fluid \
		--log=False --neighbors=2 --time_window=4 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--pretrained_model_file=$(MODEL_PATH) \
		--save_directory=$(SAVE_DIRECTORY) \
		--transformed=true

transformed_eval_tw2:
	$(PYTHON) experiments/train.py --device=cpu --experiment=fluid \
		--log=False --neighbors=1 --time_window=2 --mode=predict \
		--hidden_features=$(HIDDEN_FEATURES) \
		--pretrained_model_file=$(MODEL_PATH) \
		--save_directory=$(SAVE_DIRECTORY) \
		--transformed=true


# Test
test: install
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=fluid \
		--log=False --neighbors=1

test_param:
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=fluid \
		--time_window=$(TW) \
		--neighbors=1 \
		--hidden_features=$(HIDDEN_FEATURES) \
		--log=True

test_cpu: install_cpu
	$(PYTHON) experiments/train.py --device=cpu --experiment=fluid \
		--log=False --neighbors=1 --time_window=20

test_eval_cpu: install_cpu
	$(PYTHON) experiments/train.py --device=cpu --experiment=fluid \
		--log=False --neighbors=6 --time_window=10 --mode=predict \
		--pretrained_model_file=tests/data/pretrained/GNN_ns_fluid_n6_tw10_unrolling1_time5413650/model.pt
