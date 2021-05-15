from torch.utils.data import Dataset
from AudioClassifier.workspaces.pann_transfer.pytorch import inference as audio_classifier
from PIL import Image
from torchvision import transforms

AUDIO_FOLDER='./AudioClassifier/datasets/Sub-URMP/audio/'
IMG_FOLDER='./AudioClassifier/datasets/Sub-URMP/images/'

class sound_Image_Dataset(Dataset):

  def __init__(self, files):
    self.files = files

  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    _sound = audio_classifier.data_load(AUDIO_FOLDER + self.files[index] + '.wav')
    transform_img2 = transforms.Compose([
      transforms.Resize(128),
      transforms.CenterCrop(128),
      transforms.ToTensor()]
    )
    _image = transform_img2(Image.open(IMG_FOLDER + self.files[index] + '.jpg'))
    _tag = self.files[index].split("/")[0]
    return _sound, _image, _tag