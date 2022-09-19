# Segmentation of road surface perceptions

This was my internship project at Edge vision. The task was to build an algorithm that classifies road images based on the condition of the surface. For example, if the ratio of “snow” pixels to all pixels which belong to the road surface exceeds a certain threshold (should be parameterized), the image is classified as “snow”.

## The steps followed:
    1- Identify approach for road surface condition classification and research state of the art
    2- Identify data annotation format
    3- Prepare labeling instruction
    4- Create an evaluation dataset
    5- Label 100 training images
    6- Data preprocessing
    7- Train the prototype model
    8- Demonstrate test model run files

## Identify the approach and research state-of-the-art models
Since we have to classify the images based on the ratio of the class to all pixels in the image, image segmentation is a reasonable choice.
we started searching for previous solutions for road surface condition segmentation and found some research papers that tackle the same problem. Here is the literature review of some of the papers: https://drive.google.com/file/d/1xH8Ywu1wQfHCZDP4cPAknSNE6zbX5lEN/view?usp=sharing

## Identify data annotation format

Annotation app used: Label studio

Label interface used: Semantic segmentation with Polygons

Labels: 
- snow(0,145,225)
- wet(255,55,0)
- dry(129,96,49)
- other(141,21,239)
    
Format of the labels: COCO

## Prepare labeling instruction
Labeling instructions file: https://drive.google.com/file/d/1kU1kAjzr7eXmxId-GibMuJvmCIdaUnHD/view?usp=sharing

## Training and Evaluation datasets
Can be found here:  https://drive.google.com/drive/folders/186LubNXsJoWMrgjsFXcYtNSaRtdyTPVa?usp=sharing

## Data preprocessing
The data preprocessing step that we did was to create RGB and 1-channel images for labels of the datasets. This can be done by running the mask_generator.py file.

## Training and testing the model
The training and testing steps were done using a training pipeline tool developed by Yaroslav Shumichénko. I would like to express my gratitude to him for letting me use his tool and his guidance throughout the internship. The tool: https://github.com/Jud1cator/training-pipeline

The model used was: UNet

### Snippets from Tensorboard while training
![image](https://user-images.githubusercontent.com/71794972/191086207-46dc6276-43b1-4613-9a1c-8661a2e39244.png)
![image](https://user-images.githubusercontent.com/71794972/191086355-170b3b37-211b-403b-b3e4-5945080ce762.png)
![image](https://user-images.githubusercontent.com/71794972/191086373-4585644e-560b-4d2a-b9df-58e0d26fd4cd.png)

IoU on the testing dataset = 0.796

Description of parameters:
- As we can notice from the graphs, we achieved the highest IoU and lowest loss values when we trained the model for 40 epochs. Also, we can see from the train_loss curve that the value of the loss starts to plateau after around 35 epochs

- features_start=32 for 2 reasons:
    1- I tested the model using features_start=16 and it produce higher loss and lower IoU, which means by increasing the number of feature_starting, will give             better results

    2- I could not increase it more because I was training the model on my machine which has low GPU power 

## Example of the output
![image](https://user-images.githubusercontent.com/71794972/191086792-fa4ef267-f719-4468-be7b-008729b068cc.png)
On the left, we have the input image, on the middle we have the ground truth, and on the right, we have the model prediction for each input.

We can also have detailed information about the prediction for example:

Image number 1:
-Snow percentage 0.644
-wet percentage 0.001
-dry percentage 0.137
-other percentage 0.027
-background percentage 0.191

## Confusion matrix for the test dataset:
![image](https://user-images.githubusercontent.com/71794972/191086946-55e00c9b-212d-42bc-b3c3-9de9db0216c2.png)

From the confusion matrix, we can notice that the model confuses between the:
- wet and dry. This happens because there is a small number of wet images in the dataset 
- dry and snow. This happens when there is strong light on the road which makes it looks more like snow

These problems can be solved by collecting more images that contain both classes of confusion.

## Repository structure:

This repository is organized in a following folder structure:

- `configs` - a folder for storing training procedure configurations. Contains example config
with possible fields and values.

- `runs` - information about runs, tensorboard logs, model checkpoints and evaluation results of completed jobs.

- `src` - all source code files

The source code is organized in a following folder structure:

- `data_modules` - module which contains subclasses of the `LightningDataModule` class. Used to
perform all data related operations. Currently supports data manipulation for classification and detection tasks.

- `losses` - TBD module for custom loss functions.

- `metrics` - module which contains `AbstractMetric` class and its subclasses. These classes are meant 
as containers and aggregators of different metrics that may be collected during training procedure.

- `models` - module which contains `AbstractModelWrapper` class (a subclass of `torch.nn.Module`).
Any Pytorch neural network which is subclass of `Module` or `AbstractModelWrapper` can be added here
to be used in a training procedure.

- `tasks` - module which contains subclasses of `LightningModule` which wraps up any model from `models`
module for corresponded task, defining its training procedure.

- `utils` - all helpful unclassified code goes here

All launching scripts (like `run.py`) go to the root of `src`.

## How to install

- Clone the repository
- Create and activate the virtual environment. This is important, because you
don't want to mess with packages versions which may be incompatibe with ones you
already have in you system.
Here is how you can do it using `venv` module in Python 3:

    `python3 -m venv /path/to/new/virtual/environment`

- Install requirements:

    `pip install -r requirements.txt`

- Install `pre-commit` to automatically run `flake8` and `isort` before each commit:

    `pre-commit install`

__WARNING__: you may need to install different versions of `torch` and `torchvision`
packages depending on you CUDA version. For that, refer to the specific version
which are compatible with your CUDA version here: https://download.pytorch.org/whl/torch_stable.html
You need to __MANUALLY__ install needed version of `torch` and `torchvision`, for example
for CUDA 11.1:

    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111

## How to run

To run training pipeline, put your `yaml` config to `configs` folder and provide it to `src/train.py` script:

`python3 src/train.py -c cifar10_classification_with_simple_cnn.yml`

## Config structure

To simplify the process of training neural networks and searching the optimal hyperparameters you
can tweak all parts of training procedure in a single config file once all additionally needed features 
are implemented. It helps with tracking parameter values used for training and tuning them by 
collecting them all in one place. Below is how a sample config for running a training of image classifier
can look like:
```
run_params:
  name: example_classification
  seed: 1

datamodule:
  name: ClassificationDataModule
  params:
    data_dir: "/path/to/train/folder"
    test_data_dir: "/path/to/test/folder"
    train_split: 0.9
    val_split: 0.1
    batch_size: 32
    use_weighted_sampler: False
    pin_memory: True

train_transforms:
  - name: ToFloat
    params:
      max_value: 255
  - name: Resize
    params:
      width: 32
      height: 32
  - name: HorizontalFlip
    params:
      p: 0.5
  - name: ToTensor

val_transforms:
  - name: ToFloat
    params:
      max_value: 255
  - name: Resize
    params:
      width: 32
      height: 32
  - name: ToTensor


task:
  name: ClassificationTask
  params:
    visualize_first_batch: True
    model:
      name: EfficientNetLite0
      params:
        pretrained: True
    loss:
      name: CrossEntropyLoss
      params:
        is_weighted: False
    metrics:
      - name: F1Score
    optimizer:
      name: Adam
      params:
        lr: 0.001

callbacks:
  - name: ModelCheckpoint
    params:
      monitor: val_f1score
      mode: 'max'
      verbose: True

trainer_params:
  max_epochs: 100
  gpus: 1

export_params:
  output_name: example_classification
  to_onnx: True
```

It outlines all parameters of the training procedure: data parameters, 
transformations, model and optimizer hyperparameters, loss and metrics to collect.
Callbacks can be set to monitor the procedure, such as checkpoint monitor or early stopping.
Moreover, you can train your model on multiple GPUs by simply setting the trainer's `gpus`
parameter to the number of GPUs (Thanks to wonderful Pytorch Lightning). Finally, the trained model
can be automatically converted to ONNX format to facilitate its future deployment. ONNX can be
easily converted to such frameworks as TensorRT or OpenVINO for fast inference on GPU and CPU.


## References
<a id="1">[1]</a> 
Falcon, W., & The PyTorch Lightning team. (2019). PyTorch Lightning (Version 1.4) [Computer software]. https://doi.org/10.5281/zenodo.3828935

<a id="2">[2]</a> 
(Generic) EfficientNets for PyTorch by Ross Wightman: https://github.com/rwightman/gen-efficientnet-pytorch

<a id="3">[3]</a>
EfficientDet (PyTorch) by Ross Wightman: https://github.com/rwightman/efficientdet-pytorch

<a id="4">[4]</a>
A Notebook with sample integration of EfficientDet (PyTorch) into Pytorch Lightning:
https://gist.github.com/Chris-hughes10/73628b1d8d6fc7d359b3dcbbbb8869d7
