# Project Structure

The directory structure is as follows:

```shell
.
├── audiozen
│   ├── acoustics
│   ├── dataset
│   ├── model
│   │   ├── module
│   └── trainer
├── docs
│   └── audiozen
│       ├── acoustics
│       ├── dataset
│       └── trainer
├── notebooks
├── recipes
│   └── dns_icassp_2020
│       ├── cirm_lstm
│       ├── data
└── tools
```

- `audiozen/`: The core of the project. It contains the following subdirectories:
  - `acoustics/`: Contain the code for audio signal processing.
  - `dataset/`: Contain the data loading and processing code.
  - `model/`: Contain the code for model definition and training.
  - `trainer/`: Contain the code for training and evaluation.
  - ...
- `docs/`: Contains the project's documentation.
- `recipes/`: Contains the code for experiments. Name the subdirectory after the dataset and create a subdirectory for the dataset named after the model. For example, `dns_icassp_2020/` represents the dataset `dns_icassp_2020`, and this directory contains data loading classes, training, and inference scripts for this dataset. `cirm_lstm/` contains the model for this dataset, including the structure and trainers for each model.
- `tools/`: Contains the code for additional tools, such as data preprocessing, model conversion, etc.

For the dataset directory, take `recipes/dns_icassp_2020` as an example. Its directory structure is as follows:

```shell
.
├── cirm_lstm
│   ├── baseline.toml
│   ├── model.py
│   └── trainer.py
├── dataset_train.py
├── dataset_validation_dns_1.py
├── dataset_validation_dns_4_non_personalized.py
└── run.py
```

The scripts in the `recipes/dns_icassp_2020` directory are common to all models in this dataset, covering entry files, data loading classes, etc.:
- `run.py`: The entry of the entire project, which can be used to train all models in the `dns_icassp_2020` directory.
- `dataset_train.py`: The construction class of the training dataset.
- `dataset_validation_dns_1.py`: The construction class of the first validation dataset.
- `dataset_validation_dns_4_non_personalized.py`: The construction class of the second validation dataset.

In addition, the `recipes/dns_icassp_2020` directory can contain multiple model directories, each corresponding to a model. Each model directory contains:

- `<exp_id>.toml`: The training parameters for this model.
- `trainer.py`: The trainer for this model, which contains the operations and operations to be executed in each training, validation and test round.
- `model.py`: The structure of the current model.
- `run.py` (optional): The entry of the current model, which can be used to train the current model. If this file is not present, the `run.py` file in the `recipes/dns_icassp_2020` directory will be used.
