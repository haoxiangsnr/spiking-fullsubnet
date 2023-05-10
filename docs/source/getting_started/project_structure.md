# Project Structure

Before going to details, let's take a look at the overall structure of the project.
You may familiar with the project structure if you have used [ESPNet](https://github.com/espnet/espnet) and [SpeechBrain](https://github.com/speechbrain/speechbrain). AudioZEN is inspired by them, but it is more flexible and simpler.

AudioZEN includes a core package and a series of recipes. The core package named `audiozen` provides common audio signal processing tools and deep learning trainers. As we have installed `audiozen` in editable mode, we can call `audiozen` package in everywhere of code. In addition, we can modify the source code of `audiozen` package directly. Any changes to the original package would reflect directly in your environment. For example, we can call `audiozen` package in `recipes` folder to train models on specific datasets and call `audiozen` package in `tools` folder to preprocess data. The recipes in the `recipes` folder are used to conduct the research on the audio/speech signal processing. Recipe introduced by [Kaldi](https://kaldi-asr.org/doc/about.html) firstly. It provides a convenient and reproducible way to organize and save the deep learning training pipelines.

The directory structure is as follows:

```shell
├── 📁 audiozen
│   ├── 📁 acoustics
│   ├── 📁 dataset
│   ├── 📁 model
│   │   ├── 📁 module
│   └── 📁 trainer
├── 📁 docs
│   └── 📁 audiozen
│       ├── 📁 acoustics
│       ├── 📁 dataset
│       └── 📁 trainer
├── 📁 notebooks
├── 📁 recipes
│   └── 📁 dns_icassp_2020
│       ├── 📁 cirm_lstm
│       │   ├── 📄 baseline.toml
│       │   ├── 📄 model.py
│       │   └── 📄 trainer.py
│       ├── 📄 dataset_train.py
│       ├── 📄 dataset_validation_dns_1.py
│       ├── 📄 dataset_validation_dns_4_non_personalized.py
│       └── 📄 run.py
└── 📁 tools
```

- 📁`audiozen/`: The core of the project. It contains the following subdirectories:
    - 📁 `acoustics/`: Contain the code for audio signal processing.
    - 📁 `dataset/`: Contain the data loading and processing code.
    - 📁 `model/`: Contain the code for model definition and training.
    - 📁 `trainer/`: Contain the code for training and evaluation.
    - ...
- 📁 `docs/`: Contains the project's documentation.
- 📁 `recipes/`: Contains the recipes for specific experiments.
- 📁 `tools/`: Contains the code for additional tools, such as data preprocessing, model conversion, etc.

In `recipes` folder, we name the subdirectory after the dataset. create a subdirectory for the dataset named after the model.
For example, `recipes/dns_icassp_2020/` represents the dataset `dns_icassp_2020`, and this directory contains data loading classes, training, and inference scripts for this dataset:

- 📄 `run.py`: The entry of the entire project, which can be used to train all models in the `dns_icassp_2020` directory.
- 📄 `dataset_train.py`: The construction class of the training dataset.
- 📄 `dataset_validation_dns_1.py`: The construction class of the first validation dataset.
- 📄 `dataset_validation_dns_4_non_personalized.py`: The construction class of the second validation dataset.


`cirm_lstm/` contains the cIRM LSTM model for this dataset, including the structure and trainers for each model.

- 📄 `<exp_id>.toml`: The training parameters for this model.
- 📄 `trainer.py`: The trainer for this model, which contains the operations and operations to be executed in each training, validation and test round.
- 📄 `model.py`: The structure of the current model.
- 📄 `run.py` (optional): The entry of the current model, which can be used to train the current model. If this file is not present, the `run.py` file in the `recipes/dns_icassp_2020` directory will be used.
