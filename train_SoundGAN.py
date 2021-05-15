import os
import torch
import numpy as np

#import files
from BigGan.TFHub import converter as biggan

from LossModule.losses import PerceptualLoss
from utils import switcher
from utils import get_all_dataset_files
from utils import prep_tb
from AudioClassifier.workspaces.pann_transfer.pytorch import inference as audio_classifier
import torch.nn.functional as F
from sound_Image_Dataset import sound_Image_Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


AUDIO_FOLDER='./AudioClassifier/datasets/Sub-URMP/audio/'
IMG_FOLDER='./AudioClassifier/datasets/Sub-URMP/images/'
GENERATED_IMAGE_FOLDER = "./BigGan/TFHub/pretrained_samples/"
SOUNDGAN_CHECKPOINT_PATH='./CheckPoint/12-05-2021/'
TRANS_IN_DIM = 8192

def train(args, trans_model, date):
    #general init
    writer = SummaryWriter("./runs/" + date + "/" + args.loss_layer)
    batch_cnt = 1
    wrong_tag = 0

    #init audio classifier
    audio_model, labels, sample_rate, device = audio_classifier.init_soundGAN_train(args)

    #move Transition net to device
    trans_model.to(device)

    #init bigGAN
    bigGAN_model = biggan.convert_biggan(args.resolution, args.weights_dir, redownload=args.redownload, no_ema=args.no_ema, verbose=args.verbose)
    bigGAN_model.eval()
    bigGAN_model.to(device)

    #init Losses
    loss_percep_fn = PerceptualLoss(features_to_compute=[args.loss_layer], criterion=torch.nn.L1Loss()).to(device)

    #init optimizer
    optimizer = torch.optim.Adam(trans_model.parameters(), lr=args.lr)

    #init DataLoader
    files_paths = get_all_dataset_files()
    dataset = sound_Image_Dataset(files_paths)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    #init progress bar
    pbar = tqdm(total=((int)(len(files_paths) / args.batch_size) * args.epoch_range), desc="Training Progress: ",unit="Batch")

    for epoch in range(args.epoch_range):
        for batch_sound, batch_image, batch_tag in data_loader:
            #audio classification
            batch_waveform = audio_classifier.move_data_to_device(batch_sound, device)
            batch_audio_tag, trans_in = audio_classifier.soundGAN_audio_tag(audio_model, batch_waveform, labels)
            batch_audio_tag = np.array(batch_audio_tag)[:, 0]

            # Change audio classifier category indexes to bigGan indexes
            big_gan_class = [switcher[audio_tag_number] for audio_tag_number in batch_audio_tag]

            batch_audio_tag = np.array(labels)[batch_audio_tag]

            #check if classification is correct
            correct_tag_idx = (batch_audio_tag == batch_tag)
            correct_tag_cnt = np.sum(correct_tag_idx)
            if correct_tag_cnt == 0:
                wrong_tag += args.batch_size - correct_tag_cnt
                batch_cnt += 1
                continue
            wrong_tag += args.batch_size - correct_tag_cnt

            # remove wrong audio tags from images and audio labels (images\audio files) and move to device
            batch_image = batch_image[correct_tag_idx].to(device)

            #save tagging accuracy
            writer.add_scalar("Audio classification accuracy", 100 - (wrong_tag / (batch_cnt * args.batch_size)) * 100,  batch_cnt)
            trans_in = trans_in[correct_tag_idx] #remove incorrect tags

            #remove incorrect audio tags from bigGAN catagories
            big_gan_class = np.array(big_gan_class)[correct_tag_idx] #remove incorrect tags

            #flatten trans_in vector
            trans_in = torch.reshape(trans_in, (correct_tag_cnt, TRANS_IN_DIM))

            #run through transition net
            trans_vec = trans_model(trans_in)
            # trans_vec = torch.randn(correct_tag_cnt, 120).to(device) #TODO: sanity check

            #calculate and save mean and variance for monitoring
            writer.add_scalar("Transition Vector/Mean", torch.mean(trans_vec),  batch_cnt)
            writer.add_scalar("Transition Vector/Variance", torch.var(trans_vec),  batch_cnt)
            writer.add_histogram("Transition Vector/Histogram", trans_vec, batch_cnt)

            #generate bigGAN image
            generated_images = biggan.generate_batch(bigGAN_model, trans_vec, args.parallel, big_gan_class)

            #save batch generated images
            if batch_cnt % 100 == 0:
                with torch.no_grad():
                    writer.add_images("Batch number: " + str(batch_cnt) + "/Generated", prep_tb(generated_images))
                    writer.add_images("Batch number: " + str(batch_cnt) + "/Original", batch_image)

            with torch.set_grad_enabled(True):
                # Zero the gradients before running the backward pass.
                optimizer.zero_grad()

                # compute loss
                loss_percep_res = loss_percep_fn(generated_images, batch_image)
                loss = loss_percep_res

                #save loss
                writer.add_scalar("VGG Perceptual Loss", loss_percep_res, batch_cnt)

                # Backward pass
                loss.backward()

                optimizer.step()

                #update batch end (progress bar and batch counter)
                pbar.update(1)
                batch_cnt += 1

        #save model each epoch
        torch.save(trans_model, SOUNDGAN_CHECKPOINT_PATH + date + "/" + str(batch_cnt) + "_" + args.loss_layer + ".pth")

    pbar.close()
    writer.close()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(8192, 4096)
        self.fc2 = torch.nn.Linear(4096, 2048)
        self.fc3 = torch.nn.Linear(2048, 1024)
        self.fc4 = torch.nn.Linear(1024, 512)
        self.fc5 = torch.nn.Linear(512, 256)
        self.fc6 = torch.nn.Linear(256, 120)

    def forward(self, x):
        x.requires_grad = True
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)

        # normilize output vector to fit guassian distro of bigGAN input
        x = x/torch.sqrt(torch.var(x))
        x = x - torch.mean(x)

        return x
    