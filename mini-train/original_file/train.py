import sys
sys.path.append('..')
import json
from deepmoji.model_def import deepmoji_architecture
from train_util import class_train
from deepmoji.global_variables import NB_TOKENS, ROOT_PATH
from deepmoji.finetuning import (
    load_benchmark)

DATASET_PATH = 'raw2.pickle'
SAVE_PATH='{}/mini-train/original_file'.format(ROOT_PATH)
nb_classes=7
with open('../myx_vocabulary.json', 'r') as f:
    vocab = json.load(f)

# Load dataset.
data = load_benchmark(DATASET_PATH, vocab)

# Set up model and train
model = deepmoji_architecture(2,NB_TOKENS,data['maxlen'])
model.summary()
model, f1 = class_train(model, data['texts'], data['labels'], nb_classes,
                      data['batch_size'], method='new',savepath=SAVE_PATH,
                      epoch_size=256,nb_epochs=100)
model.save_weights('model_epoch64.hdf5')
print('F1: {}'.format(f1))
