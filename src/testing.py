import io
import logging
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from metrics.iou_metric import IoUMetric

from registry import Registry
from utils.config_validation import Config
from utils.helpers import create_config_parser
import matplotlib.pyplot as plt
from typing import List, Optional, Union
from torchvision.utils import draw_bounding_boxes, make_grid
from torch.utils.tensorboard import SummaryWriter
from metrics.confusion_matrix import ConfusionMatrix

classes_perecentages = []


def label_to_color(labels,output_path):
    colored_labels=[]
    batch_size, w, h = labels.shape
    idx_to_color = {
        '0': [0,0,0],
        '1': [0,55,255],
        '2': [49,96,129],
        '3': [239,21,141],
        '4': [225, 145, 0]
    }
    for i in range(batch_size):
        img = labels[i].numpy()
        colored_label = np.zeros((w,h,3))
        classes = np.zeros(5)
        lenght = w*h
        for j in range(w):
            for k in range(h):
                colored_label[j][k][0]=idx_to_color[str(img[j][k])][0]
                colored_label[j][k][1]=idx_to_color[str(img[j][k])][1]
                colored_label[j][k][2]=idx_to_color[str(img[j][k])][2]
                classes[img[j][k]] += (1/lenght)
        classes_perecentages.append(classes)
        cv2.imwrite(output_path + str(i) +".png",colored_label)
        colored_labels.append(colored_label)
    return colored_labels


def print_grid(imgs,masks,output):
    fig = plt.figure(figsize=(5.,5.))
    grid = ImageGrid(fig, 111, nrows_ncols=(5,3),axes_pad=0.1)
    imgs_list = []
    for i in range(5):
        im1 = imgs[i].numpy()
        im1 = np.transpose(im1, (1, 2, 0))
        imgs_list.append(im1)
        imgs_list.append(masks[i].astype('uint8'))
        imgs_list.append(output[i].astype('uint8'))
    for ax, im in zip(grid,imgs_list):
        ax.imshow(im)
    plt.show()
def print_info(preds,masks):
    total_cm = np.zeros((5,5))
    for i in range(preds.shape[0]):
        print(f"Image number {i}: ")
        print(f"-Snow percentage "+ format(classes_perecentages[i][0],".3f"))
        print(f"-wet percentage "+ format(classes_perecentages[i][1],".3f"))
        print(f"-dry percentage "+ format(classes_perecentages[i][2],".3f"))
        print(f"-other percentage "+ format(classes_perecentages[i][3],".3f"))
        print(f"-background percentage "+ format(classes_perecentages[i][4],".3f"))
        total_cm += confusion_matrix(preds[i],masks[i])
    print("Confusion matrix for the test dataset:")
    print(total_cm)

def confusion_matrix(input, target):
    cm = ConfusionMatrix(5)
    cm.update(input,target)
    return cm.get_value()

def prepare_run(name, seed):
    pl.seed_everything(seed)
    all_runs_dir = Path('./runs')
    date = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    run_name = '_'.join([str(date), f'_{name}'])
    run_dir = all_runs_dir / run_name
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    test_res_dir = run_dir / 'results'
    test_res_dir.mkdir(exist_ok=True, parents=True)
    weights_dir = run_dir / 'trained_models'
    weights_dir.mkdir(exist_ok=True, parents=True)
    return run_dir, checkpoints_dir, test_res_dir, weights_dir
def main(
        run_params: dict,
        datamodule: dict,
        train_transforms: list,
        val_transforms: list,
        task: dict,
        callbacks: list,
        trainer_params: dict,
        export_params: dict,
        test_transforms: list
):
    # Scanning src module to fill the registry
    Registry.init_modules()

    # Initializing transforms
    test_tf_list = []
    for tf in test_transforms:
        transform = Config(**tf)
        test_tf_list.append(Registry.TRANSFORMS[transform.name](**transform.params))

    # Initializing datamodule
    datamodule_config = Config(**datamodule)
    dm = Registry.DATA_MODULES[datamodule_config.name](
        **datamodule_config.params,
        train_transforms=test_tf_list,
        val_transforms=test_tf_list,
    )

    # Initializing task
    task_config = Config(**task)
    task = Registry.TASKS[task_config.name](datamodule=dm, **task_config.params)

    # loading the pretrained model
    model_path = r"C:\Users\Electronica\Edge vision internship\prototype\Training Pipeline\saved_models\best.pt"
    task.model.load_state_dict(torch.load(model_path))
    
    # --- EVALUATION ---
    task.eval()
    val_data = dm.test_dataloader()
    imgs, masks = next(iter(val_data))
    output = task(imgs)
    output = torch.argmax(output,dim=1) # shape = [batch_size, width, hight]
    iou = IoUMetric(5)
    iou.update(output,masks)
    output_colored = label_to_color(output,"./data/test_pred/")
    masks_colored = label_to_color(masks,"./data/test_labels/")

    # printing the results
    print_grid(imgs,masks_colored,output_colored)
    print_info(output,masks)
    print(iou.get_value())
    

if __name__ == '__main__':
    _, config = create_config_parser()
    main(**config)
