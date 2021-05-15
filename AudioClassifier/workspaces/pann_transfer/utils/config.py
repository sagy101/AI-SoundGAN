sample_rate = 44000
#TODO: might be wrong to pad zeroes to 1 sec
clip_samples = sample_rate * 1

mel_bins = 64
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None
fmin = 50
fmax = 14000
window_size = 1024
hop_size = 320

labels = ['bassoon', 'cello', 'clarinet', 'double_bass', 'flute', 'horn', 'oboe',
    'sax', 'trombone', 'trumpet', 'tuba', 'viola', 'violin']
    
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)