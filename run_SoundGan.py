
import os
import torch

#import files
from BigGan.TFHub import converter as biggan
from AudioClassifier.workspaces.pann_transfer.pytorch import inference as audio_classifier
from utils import switcher
from utils import imageNet_labels
from utils import parse_args
from utils import prep_audio
from train_SoundGAN import train
from train_SoundGAN import Net
from LossModule.diagnose import calc_loss
from LossModule.diagnose import diagnose_loss
from datetime import datetime
from LossModule.diagnose import classify_images

#define constants
AUDIO_CHECKPOINT_PATH='./AudioClassifier/workspaces/pann_transfer/checkpoints/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/10000_iterations.pth'
AUDIO_PATH='./AudioClassifier/datasets/Sub-URMP/audio/'
IMG_PATH='./AudioClassifier/datasets/Sub-URMP/images/'
CLASSIFICATION_RESULTS_DIR = "./LossModule/results/classification/"
IMG_TEST='./TestData/Images/'
AUDIO_TEST='./TestData/Audio/'
GENERATED_TEST='./TestData/Generated/'
INSTRUMENT_PATH='cello/cello00_500'
MODEL_TYPE='Transfer_Cnn14'
SOUNDGAN_CHECKPOINT_PATH='./CheckPoint/12-05-2021/'
LAYERS = ['relu3_1', 'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4', 'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4']

DEVICE = 'cuda'


def audio_tag(args):
  audio_classifier.audio_tagging(args)

def bigGAN(args, z = 0):
  # Running GAN
  os.makedirs(args.weights_dir, exist_ok=True)
  os.makedirs(args.samples_dir, exist_ok=True)
  if args.resolution is not None:
    print("res: " + str(args.resolution))
    G = biggan.convert_biggan(args.resolution, args.weights_dir, redownload=args.redownload, no_ema=args.no_ema,
                              verbose=args.verbose)
    if args.generate_samples:
      original_filename = os.path.join(args.samples_dir, f'biggan{args.resolution}_') + imageNet_labels[args.label][0]
      print('Generating samples...')
      if z == 0:
        z = torch.randn(args.samples, G.dim_z).to(DEVICE)
      biggan.generate_sample(G, z, biggan.Z_DIMS[args.resolution], args.batch_size, original_filename, args.parallel,
                             args.label, args.samples, args.mode)

def generate(args):
  for audio_file in os.listdir(AUDIO_TEST):
    # Audio tagging
    audio_file = prep_audio(audio_file)
    audio_tag_info = audio_classifier.audio_tagging(args, audio_file=AUDIO_TEST + audio_file)
    audio_tag_name = audio_tag_info[1][audio_tag_info[2][0]]
    trans_in = audio_tag_info[3]
    # Change audio_tag category indexes to bigGan indexes
    big_gan_class = switcher[audio_tag_info[2][0]]

    if big_gan_class == -1:
      print("No such class " + audio_tag_info[1][audio_tag_info[2][0]] + " in imageNet.")
      continue  # error
    print("Sound label is: " + imageNet_labels[big_gan_class])
    #transition net
    model = torch.load(SOUNDGAN_CHECKPOINT_PATH + 'final_relu5_1.pth')
    model.eval()
    trans_in = torch.reshape(trans_in, (1, 8192))
    trans_vec = model(trans_in)

    # Running GAN
    G = biggan.convert_biggan(args.resolution, args.weights_dir, redownload=args.redownload, no_ema=args.no_ema,
                              verbose=args.verbose)
    if args.generate_samples:
      generated_file_path = os.path.join(args.generated_dir, f'biggan{args.resolution}_') + audio_file[:-4]
      print('Generating samples...')
      biggan.generate_sample(G, trans_vec, biggan.Z_DIMS[args.resolution], args.batch_size, generated_file_path, args.parallel,
                             big_gan_class, args.samples, args.mode)

    #loss
    calc_loss(img1=generated_file_path+'.jpg', img2=IMG_TEST + audio_file[:-4]+'.jpg')

def train_loss_diagnose(date):
  if os.path.isfile(SOUNDGAN_CHECKPOINT_PATH + date):
    os.mkdir(SOUNDGAN_CHECKPOINT_PATH + date)
  for layer in LAYERS:
    net = Net()
    args.loss_layer = layer
    train(args, net, date)
    torch.save(net, SOUNDGAN_CHECKPOINT_PATH + date + "/final_" + layer + ".pth")

if __name__ == '__main__':
  args = parse_args()

  date = datetime.today().strftime('%d-%m-%Y')

  if args.mode == "audio_tagging":
    audio_tag(args)
  elif args.mode == "bigGAN":
    bigGAN(args)
  elif args.mode == "loss":
    calc_loss(args.img1, args.img2)
  elif args.mode == "train":
    if os.path.isfile(SOUNDGAN_CHECKPOINT_PATH + date):
      os.mkdir(SOUNDGAN_CHECKPOINT_PATH + date)
    net = Net()
    train(args, net, date)
    torch.save(net, SOUNDGAN_CHECKPOINT_PATH + date + "/final.pth")
  elif args.mode == 'generate':
    generate(args)
  elif args.mode == "train_loss_diagnose":
    train_loss_diagnose(date)
  elif args.mode == "calc_loss_diagnose":
    diagnose_loss(LAYERS, date)
  elif args.mode == "loss_diagnose":
    train_loss_diagnose(date)
    diagnose_loss(LAYERS, date)
  elif args.mode == "classify_image":
    if os.path.isfile(CLASSIFICATION_RESULTS_DIR):
      os.mkdir(CLASSIFICATION_RESULTS_DIR)
    classify_images(date)

