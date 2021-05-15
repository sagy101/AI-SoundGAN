import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa


from AudioClassifier.workspaces.pann_transfer.pytorch.models import *
from AudioClassifier.workspaces.pann_transfer.pytorch.pytorch_utils import move_data_to_device
import AudioClassifier.workspaces.pann_transfer.utils.config as config


def audio_tagging(args, audio_file, mode='tag'):
    """Inference audio tagging result of an audio clip.
    """
    # Arugments & parameters
    print("Tagging audio file: " + audio_file)
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = audio_file
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num, freeze_base=True)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)
    trans_in = batch_output_dict[1]
    batch_output_dict = batch_output_dict[0]
    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]

    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    # if mode == 'tag':
    #     for k in range(10):
    #         print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
    #             clipwise_output[sorted_indexes[k]]))

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        # print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels, sorted_indexes, trans_in


def init_soundGAN_train(args):
    """Inference audio tagging result of an audio clip.
    """
    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num, freeze_base=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    return model, labels, sample_rate, device

def data_load(audio_path, sample_rate=48000):
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    # waveform = waveform[None, :]  # (1, audio_length)
    return waveform

def soundGAN_audio_tag(model, waveform, labels):
    # Forward
    with torch.set_grad_enabled(True):
        model.eval()
        batch_output_dict = model(waveform, None)
    trans_in = batch_output_dict[1]
    batch_output_dict = batch_output_dict[0]
    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()

    """(classes_num,)"""

    sorted_indexes = np.flip(np.argsort(clipwise_output, axis=1), axis=1)

    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]

    return sorted_indexes, trans_in


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=32000)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000) 
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)

    else:
        raise Exception('Error argument!')