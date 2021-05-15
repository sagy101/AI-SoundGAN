import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

IMAGES_PATH = "./AudioClassifier/datasets/Sub-URMP/images/"


def prep_img_gen(img1, img2):
    transform_img = transforms.Compose([
      transforms.Resize(128),
      transforms.CenterCrop(128),
      transforms.ToTensor()]
    )
    _target = transform_img(Image.open(img2))
    _input = transform_img(Image.open(img1))
    return _input, _target


def prep_img(img):
    transform_img = transforms.Compose([
      transforms.Resize(128),
      transforms.CenterCrop(128),
      transforms.ToTensor()]
    )
    _img = transform_img(Image.open(img))
    _img = _img[None, :, : , :]
    return _img


def shave_edge(x, shave_h, shave_w):
    return F.pad(x, [-shave_w, -shave_w, -shave_h, -shave_h])


def get_dataset_files(Class, flag_same, dilute):
    files = []
    i = 0
    starting_dilute = dilute
    for instrument_folder in os.listdir(IMAGES_PATH):
        same_class = Class in instrument_folder
        if (same_class and flag_same == 'different') or (not same_class and flag_same == 'same'):
            continue
        instrument_images = os.listdir(IMAGES_PATH + instrument_folder)
        if len(instrument_images)/dilute < 60:
            dilute = (int)(len(instrument_images)/60)
        for image in instrument_images:
            if i % dilute == 0:
                files.append(IMAGES_PATH + instrument_folder + '/' + image)
            i += 1
        dilute = starting_dilute
    return files


def get_all_loss_class_files(Class):
    loss_files_path = "./LossModule/Dataset/"
    files = []
    for image in os.listdir(loss_files_path + Class):
        files.append(loss_files_path + Class + '/' + image)
    return files