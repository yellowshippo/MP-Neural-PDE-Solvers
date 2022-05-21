CUDA = cu111  # CUDA version 11.1
PYTHON ?= poetry run python3
PIP ?= pip3


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


# Prediction

predict_tw20: install
	OMP_NUM_THREADS=1 $(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=11 --time_window=20 --mode=predict \
		--pretrained_model_file=pretrained/GNN_ns_fluid_n11_tw20_unrolling1_time5413647/model.pt

predict_tw10: install
	OMP_NUM_THREADS=1 $(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=6 --time_window=10 --mode=predict \
		--pretrained_model_file=pretrained/GNN_ns_fluid_n6_tw10_unrolling1_time5413650/model.pt

predict_tw4: install
	OMP_NUM_THREADS=1 $(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=2 --time_window=4 --mode=predict \
		--pretrained_model_file=pretrained/GNN_ns_fluid_n2_tw4_unrolling1_time5413654/model.pt

predict_tw2: install
	OMP_NUM_THREADS=1 $(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=1 --time_window=2 --mode=predict \
		--pretrained_model_file=pretrained/GNN_ns_fluid_n1_tw2_unrolling1_time5413659/model.pt

transformed_predict_tw20: install
	$(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=11 --time_window=20 --mode=predict \
		--pretrained_model_file=pretrained/GNN_ns_fluid_n11_tw20_unrolling1_time5413647/model.pt \
		--transformed=true

transformed_predict_tw10: install
	$(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=6 --time_window=10 --mode=predict \
		--pretrained_model_file=pretrained/GNN_ns_fluid_n6_tw10_unrolling1_time5413650/model.pt \
		--transformed=true

transformed_predict_tw4: install
	$(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=2 --time_window=4 --mode=predict \
		--pretrained_model_file=pretrained/GNN_ns_fluid_n2_tw4_unrolling1_time5413654/model.pt \
		--transformed=true

transformed_predict_tw2: install
	$(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=1 --time_window=2 --mode=predict \
		--pretrained_model_file=pretrained/GNN_ns_fluid_n1_tw2_unrolling1_time5413659/model.pt \
		--transformed=true


# Learning

f_tw20: install
	$(PYTHON) experiments/flearn.py --device=cuda:0 --experiment=fluid \
		--time_window=20 \
		--neighbors=11 \
		--hidden_features=128 \
		--log=True

f_tw10: install
	$(PYTHON) experiments/flearn.py --device=cuda:0 --experiment=fluid \
		--time_window=10 \
		--neighbors=6 \
		--hidden_features=128 \
		--log=True

f_tw4: install
	$(PYTHON) experiments/flearn.py --device=cuda:0 --experiment=fluid \
		--time_window=4 \
		--neighbors=2 \
		--hidden_features=128 \
		--log=True

f_tw2: install
	$(PYTHON) experiments/flearn.py --device=cuda:0 --experiment=fluid \
		--time_window=2 \
		--neighbors=1 \
		--hidden_features=128 \
		--log=True


# Test

we3_cpu: install
	$(PYTHON) experiments/train.py --device=cpu --experiment=WE3 \
		--base_resolution=250,40 --neighbors=1 --time_window=25

we3: install
	$(PYTHON) experiments/train.py --device=cuda:0 --experiment=WE3 \
		--base_resolution=250,40 --neighbors=1 --time_window=25

test: install
	$(PYTHON) experiments/flearn.py --device=cuda:0 --experiment=fluid \
		--log=False --neighbors=1

test_cpu: install
	$(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=1 --time_window=20

test_predict_cpu: install
	$(PYTHON) experiments/flearn.py --device=cpu --experiment=fluid \
		--log=False --neighbors=6 --time_window=10 --mode=predict \
		--pretrained_model_file=tests/data/pretrained/GNN_ns_fluid_n6_tw10_unrolling1_time5413650/model.pt

flearn: install
	$(PYTHON) experiments/flearn.py --device=cuda:0 --experiment=fluid \
		--log=False --neighbors=4
