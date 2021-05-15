#!/usr/bin/env python3
BACKUP_DIR = '/home/sagy.gersh@staff.technion.ac.il/Backup'
PROJECT_DIR = '/home/sagy.gersh@staff.technion.ac.il/SoundGan'

from shutil import copytree, ignore_patterns

copytree(PROJECT_DIR, BACKUP_DIR, ignore=ignore_patterns('runs', '.git', 'core', '*.wav', '*.h5', '*.pth', '*.jpg', 'venv', '*.png', '*.log', '*.pkl', '*.pickle'))
