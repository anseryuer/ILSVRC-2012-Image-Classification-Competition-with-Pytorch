# ILSVRC 2012 Image Classification Competition with Pytorch

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

 - Completing the classification task on the ILSVRC dataset with pytorch from ground up.
 - Containing the dataset preprocess, loading, training, validation and testing part, offering customization ability in model structure, data transformation and other parts in the task. 
 - Implemented with PyTorch, can be used with GPU acceleration.

 - Dataset: [Imagenet Large Scale Visual Recognition Competition](https://image-net.org/challenges/LSVRC/2012)

 - The Jupyter Notebook version of the whole project can be accessed on [my Kaggle](https://www.kaggle.com/code/tianbaiyutoby/ilsvrc-classification-v6). Still working on it.

## Table of Contents

- [ILSVRC 2012 Image Classification Competition with Pytorch](#ilsvrc-2012-image-classification-competition-with-pytorch)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Program Folder Structure](#program-folder-structure)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Program Folder Structure

```
ILSVRC 2012 Image Classification Competition with Pytorch/
├── README.md         # Project description and usage instructions
├── dataset/          # Not included here, need to be downloaded from the dataset link above
│   ├── ILSVRC2012_devkit_t12/ 
|   |   ├── COPYING
|   |   ├── data
|   |   |   ├── ILSVRC2012_validation_ground_truth.txt
|   |   |   └── meta.mat
|   |   ├── evaluation       # This doesn't really matter here
|   |   |   └── ...
|   |   └── readme.txt       # Dataset information
│   ├── ILSVRC2012_img_train/        # Folder for training data (all 1000 categories of images)
│   │   ├── n01440764/ # Each folder
|   │   │   ├── n01440764_78.JPEG
│   │   │   └── ...
│   │   └── ...
│   ├── ILSVRC2012_img_val/   # Folder for validation data
│   │   ├── ILSVRC2012_val_00000001.JPEG
│   │   └── ...
│   └── test/         # Folder for test data
│       ├── ILSVRC2012_test_00000001.JPEG
│       └── ...
├── models/
│   ├── model.py       # Script defining the deep learning model architecture
│   └── ONNX/   # Folder to store saved model ONNX
|       └── export.py  # Export model to ONNX
├── utils/
│   ├── data_utils.py  # Script for data loading, preprocessing, and augmentation
│   ├── training.py    # Script for training the deep learning model
│   ├── plot.py        # Script for ploting image in torch.Tensor 
│   ├── transforms.py  # Script for data transformation and preprocessing
│   └── evaluation.py  # Script for evaluating the trained model and output the final result.
├── requirements.txt  # List of required Python libraries
└── train.py           # Main script to run the training process
```

## Usage

To use this project, follow these steps:

1. Download the ILSVRC dataset from [ILSVRC 2012 Image Competition](https://image-net.org/challenges/LSVRC/2012).
2. Train the classification model: `python train.py`
   
## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -am 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
