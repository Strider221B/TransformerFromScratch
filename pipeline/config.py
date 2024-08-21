from pathlib import Path

from constants import Constants as const

class Config:

    def get_config():
        return {
            const.BATCH_SIZE: 8,
            const.NO_OF_EPOCHS: 20,
            const.LEARNING_RATE: 10**-4,
            const.SEQUENCE_LENGTH: 350,
            const.MODEL_DIMENSION: 512,
            const.LANGUAGE_SOURCE: 'en',
            const.LANGUAGE_TARGET: 'it',
            const.MODEL_FOLDER: 'weights',
            const.MODEL_BASENAME: 'tmodel_',
            const.PRELOAD: None,
            const.TOKENIZER_FILE: 'tokenizer_{0}.json',
            const.EXPERIMENT_NAME: 'runs/tmodel'
        }
    
    def get_weights_file_path(config, epoch: str):
        model_folder = config[const.MODEL_FOLDER]
        model_basename = config[const.MODEL_BASENAME]
        model_filename = f'{model_basename}{epoch}.pt'
        return str(Path('.') / model_folder / model_filename)