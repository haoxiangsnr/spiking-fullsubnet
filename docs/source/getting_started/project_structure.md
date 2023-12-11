# Project Structure

Before going into details, let's take a look at the overall structure of the project.
You may familiar with this project structure (`recipes/<dataset>/<model>`) if you have used [ESPNet](https://github.com/espnet/espnet) and [SpeechBrain](https://github.com/speechbrain/speechbrain) before.
AudioZEN is inspired by them, but it is more simpler.

AudioZEN includes a core package (`/audiozen`) and a series of training recipes (`/recipes`).
The core package is named `audiozen`, which provides common audio signal processing tools and deep learning trainers. As we have installed `audiozen` in editable mode, we can call `audiozen` package everywhere in the code. In addition, we can modify the source code of `audiozen` package directly. Any changes to the original package would reflect directly in your environment. For example, we can call `audiozen` package in `recipes` folder to train models on specific datasets and call `audiozen` package in `tools` folder to preprocess data. The recipes in the `recipes` folder are used to research the audio/speech signal processing. The recipe concept was introduced by [Kaldi](https://kaldi-asr.org/doc/about.html) first, providing a convenient and reproducible way to organize and save the deep learning training pipelines.

The directory structure is as follows:

```shell
├── audiozen/
│   ├── acoustics/
│   ├── dataset/
│   ├── model/
│   │   ├── module/
│   └── trainer/
├── docs/
├── notebooks/
├── recipes/
│   └── intel_ndns/
│       ├── sdnn_delays/
│       │   ├── baseline.toml
│       │   ├── model.py
│       │   └── trainer.py
│       ├── dataloader.py
│       ├── loss.py
│       └── run.py
└── tools/
```

- `audiozen/`: The core of the project. After installing `audiozen` in the editable mode, we can call `audiozen` package everywhere in the project.
    - `acoustics/`: Contain the code for audio signal processing.
    - `dataset/`: Contain the data loading and processing code.
    - `model/`: Contain the code for model definition and training.
    - `trainer/`: Contain the code for training and evaluation.
    - ...
- `docs/`: Contains the project's documentation. We use [Sphinx Documentation Generator](https://www.sphinx-doc.org/en/master/) to build the documentation.
- `recipes/`: Contains the recipes for specific experiments. It follows a `<dataset_name>/<model_name>` structure.
- `tools/`: Contains the code for additional tools, such as data preprocessing, model conversion, etc.

In the `recipes` folder, we name the subdirectory after the dataset. create a subdirectory for the dataset named after the model.
For example, `recipes/intel_ndns/` saves the models trained on the Intel Neuromorphic DNS Challenge dataset. It contains commonly-used data loading classes, training, and inference scripts.

- `run.py`: The entry of the entire project, which can be used to train and evaluate all models in the `intel_ndns` directory.
- `dataloader.py`: The data loading and processing code for the Intel Neuromorphic DNS Challenge dataset.
- `loss.py`: The loss function commonly used in the Intel Neuromorphic DNS Challenge dataset.


`sdnn_delays/` contains the SDNN baseline model for this dataset, including the model structure and trainer:

- `<exp_id>.toml`: The training configuration file for this model.
- `trainer.py`: The trainer for this model, which contains the operations and operations to be executed in each training, validation and test round.
- `model.py`: The structure of the current model.
- `run.py` (optional): The entry of the current model, which can be used to train the current model. If this file is not present, the `run.py` file in the `recipes/intel_ndns` directory will be used.
