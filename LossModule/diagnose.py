import torch
from LossModule.losses import PerceptualLoss
from LossModule.misc import prep_img
from LossModule.misc import get_dataset_files
from LossModule.misc import get_all_loss_class_files
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from utils import imageNet_labels
import os
from utils import get_all_dataset_files

CLASSIFICATION_RESULTS_DIR = "./LossModule/results/classification/loss/"
IMG_FOLDER = './AudioClassifier/datasets/Sub-URMP/images/'
DEVICE = 'cuda'


def calc_loss(img1,img2):
  target = prep_img(img1)
  input = prep_img(img2)
  loss_fn = PerceptualLoss(features_to_compute=['relu3_1'], criterion=torch.nn.L1Loss())#TODO: change to selected layer
  loss = loss_fn(input, target)
  print("LOSS between image " + img1 + "and image " + img2 + ' is: ' + str(loss))
  print("")


def get_loss(loss_files_paths, dataset_files_paths, pbar, loss_fn):
    loss = []
    for dataset_file in dataset_files_paths:
        dataset_file = prep_img(dataset_file).to(DEVICE)
        for loss_file in loss_files_paths:
            loss_file = prep_img(loss_file).to(DEVICE)
            loss.append(loss_fn(loss_file, dataset_file))
            pbar.update(1)
    return loss


def diagnose_loss(layers, date):

    different_class_files_paths = get_dataset_files("cello", "different", 100)
    same_class_files_paths = get_dataset_files("cello", "same", 60)
    loss_class_files_paths = get_all_loss_class_files("cello")

    # init progress bar
    total_loss = len(layers)*(len(different_class_files_paths) + len(same_class_files_paths))*len(loss_class_files_paths)
    pbar = tqdm(total=total_loss, desc="loss compute: ", unit="loss")


    loss_different_class_mean = []
    loss_same_class_mean = []

    for i, layer in enumerate(layers):
        #calculate losses of same and diffrent classes
        loss_fn = PerceptualLoss(features_to_compute=[layer], criterion=torch.nn.L1Loss()).to(DEVICE)
        loss_different_class = get_loss(loss_class_files_paths, different_class_files_paths, pbar, loss_fn)
        loss_same_class = get_loss(loss_class_files_paths, same_class_files_paths, pbar, loss_fn)

        #calculate mean of those losses
        loss_different_class_mean.append(sum(loss_different_class)/len(loss_different_class))
        loss_same_class_mean.append(sum(loss_same_class)/len(loss_same_class))

        #write layer results to file
        file = open("./LossModule/results/loss/" + date + ".txt", "a")
        file.write(layers[i] + " results:\n")
        file.write("different loss: " + str(loss_different_class_mean[i]) + "\n")
        file.write("same loss: " + str(loss_same_class_mean[i]) + "\n")
        file.write("loss ratio: " + str(loss_different_class_mean[i]/loss_same_class_mean[i]) + "\n")
        file.write("\n\n")
        file.close()

        i += 1

    pbar.close()


def classify_images(date):
    total_acc = len(get_all_dataset_files())
    pbar = tqdm(total=total_acc, desc="accuracy compute: ", unit="acc_calcs")

    #init
    vgg19 = models.vgg19(pretrained=True)
    vgg19.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #calculate accuracy over all classes
    accuracy = [0]*len(os.listdir(IMG_FOLDER))
    for i, instrument in enumerate(os.listdir(IMG_FOLDER)):
        for image in os.listdir(IMG_FOLDER + instrument):
            input_image = Image.open(IMG_FOLDER + instrument + "/" + image)
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                vgg19.to('cuda')

            with torch.no_grad():
                output = vgg19(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            classification = imageNet_labels[torch.argmax(probabilities).item()]
            if instrument in classification:
                accuracy[i] += 1
            pbar.update(1)
        accuracy[i] /= len(os.listdir(IMG_FOLDER + instrument))
    file = open(CLASSIFICATION_RESULTS_DIR + date + ".txt", "a")
    for i, instrument in enumerate(os.listdir(IMG_FOLDER)):
        file.write("class " + instrument + ": " + str(accuracy[i]) + "\n")
    file.close()
    pbar.close()
