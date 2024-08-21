import warnings
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm

from components.transformer import Transformer
from constants import Constants as const
from pipeline.config import Config
from pipeline.bilingual_dataset import BilingualDataset
from translation_model import TranslationModel

class Trainer:

    @classmethod
    def train_model(cls, config):
        device = torch.device(const.CUDA if torch.cuda.is_available() else const.CPU)
        print(f'Using device: {device}')

        Path(config[const.MODEL_FOLDER]).mkdir(parents=True, exist_ok=True)
        train_dataloader, validation_dataloader, tokenizer_source, tokenizer_target = cls._get_dataset(config)
        model = cls._get_model(config, 
                               tokenizer_source.get_vocab_size(), 
                               tokenizer_target.get_vocab_size()).to(device)
        writer = SummaryWriter(config[const.EXPERIMENT_NAME])
        optimizer = torch.optim.adam.Adam(model.parameters(), lr=config[const.LEARNING_RATE], eps=1e-9)

        initial_epoch = 0
        global_step = 0

        if config[const.PRELOAD]:
            model_filename = Config.get_weights_file_path(config, config[const.PRELOAD])
            print(f'Preloading model: {model_filename}')
            state = torch.load(model_filename)
            initial_epoch = state[const.EPOCH] + 1
            optimizer.load_state_dict(state[const.OPTIMIZER_STATE_DICT])
            global_step = state[const.GLOBAL_STEP]
        
        # Label smoothing - we ask the model to be less confident about its prediction. Whatever is the
        # result with highest prediction, we take the fraction mentioned in the label_smoothing parameter
        # and distribute it to the other predictions.
        loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id[const.TOKEN_PADDING],
                                            label_smoothing=0.1).to(device)
        
        for epoch in range(initial_epoch, config[const.NO_OF_EPOCHS]):
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch: 02d}')
            for batch in batch_iterator:

                model.train() # Since we will also be running validation in this loop, we reset to train at start.

                encoder_input = batch[const.INPUT_ENCODER].to(device) # (Batch, sequence length)
                decoder_input = batch[const.INPUT_DECODER].to(device) # (Batch, sequence length)
                encoder_mask = batch[const.MASK_ENCODER].to(device)  # (Batch, 1, 1, sequence length)
                decoder_mask = batch[const.MASK_DECODER].to(device) # (Batch, 1, sequence length, sequence length)

                encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, sequence length, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch, sequence length, d_model)
                projection_output = model.project(decoder_output) # (Batch, sequence length, target vocab size)

                label = batch[const.LABEL].to(device) # (Batch, Sequence length)

                 # (Batch, sequence length, target_vocab_size) -> (Batch * sequence length, target_vocab_size)
                loss = loss_function(projection_output.view(-1, tokenizer_target.get_vocab_size()),
                                     label.view(-1))
                batch_iterator.set_postfix(f'Loss: {loss.item(): 6.3f}')
                
                writer.add_scalar(const.TRAIN_LOSS. loss.item(), global_step)
                writer.flush()

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % 5 == 0:
                    cls._run_validation(model,
                                        validation_dataloader,
                                        tokenizer_source,
                                        tokenizer_target,
                                        config[const.SEQUENCE_LENGTH],
                                        device,
                                        lambda msg: batch_iterator.write(msg),
                                        global_step,
                                        writer) # 23848

            model_filename = Config.get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

    @classmethod
    def _run_validation(cls,
                        model: Transformer, 
                        validation_dataset, 
                        tokenizer_src: Tokenizer, 
                        tokenizer_tgt: Tokenizer, 
                        max_len,
                        device,
                        print_msg,
                        global_state,
                        writer,
                        num_examples=2):
        model.eval()
        count = 0

        source_texts = []
        expected = []
        predicted = []

        console_width = 80

        with torch.no_grad():
            for batch in validation_dataset:
                count += 1
                encoder_input = batch[const.INPUT_ENCODER].to(device)
                encoder_mask = batch[const.MASK_ENCODER].to(device)
                decoder_input = batch[const.INPUT_DECODER].to(device)

                assert encoder_input.size(0) == 1, 'Batch size must be one for validation.'

                model_output = cls._greedy_decode(model, 
                                                  encoder_input, 
                                                  encoder_mask,
                                                  tokenizer_src,
                                                  tokenizer_tgt,
                                                  max_len,
                                                  device)
                
                source_texts.append(batch[const.TEXT_SOURCE][0])
                expected.append(batch[const.TEXT_TARGET][0])
                predicted.append(tokenizer_tgt.decode(model_output.detach().cpu().numpy()))

                # If we had used print directly then it would have interfered with tqdm progress bar.
                print_msg('-' * console_width)
                print_msg(f'Source: {source_texts[-1]}')
                print_msg(f'Target: {expected[-1]}')
                print_msg(f'Predicted: {predicted[-1]}')

                if count == num_examples:
                    break

    
    @classmethod
    def _greedy_decode(cls,
                       model: Transformer, 
                       source,
                       source_mask,
                       tokenizer_src: Tokenizer, 
                       tokenizer_tgt: Tokenizer, 
                       max_len,
                       device):
        sos_index = tokenizer_src.token_to_id([const.TOKEN_START_OF_SENTENCE])
        eos_index = tokenizer_src.token_to_id([const.TOKEN_END_OF_SENTENCE])

        encoder_output = model.encode(source, source_mask)
        # (1, 1) in the torch empty is because of batch and the token of the decoder input.
        decoder_input = cls._get_decoder_token_for(sos_index, source, device)
        next_word = None
        while ((decoder_input.size(1) < max_len) and (next_word != eos_index)):
            decoder_mask = BilingualDataset.get_causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            probability = model.project(output[:, -1]) # we need the projection of the last token
            _, next_word = torch.max(probability, dim=1) # greedy search
            decoder_input = torch.concat([decoder_input, 
                                          cls._get_decoder_token_for(next_word.item(), source, device)])
        
        return decoder_input.squeeze(0) # removing batch dimension.


    @staticmethod
    def _get_decoder_token_for(value, type_as, device):
        return torch.empty(1, 1).fill_(value).type_as(type_as).to(device)

    @staticmethod
    def _get_model(config: dict, vocab_src_len: int, vocab_trgt_len: int):
        model = TranslationModel.build_transformer(vocab_src_len, 
                                                   vocab_trgt_len,
                                                   config[const.SEQUENCE_LENGTH],
                                                   config[const.SEQUENCE_LENGTH],
                                                   config[const.MODEL_DIMENSION])
        return model

    @classmethod
    def _get_dataset(cls, config: dict) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
        raw_dataset = load_dataset(const.DATASET_NAME, 
                                   f'{config[const.LANGUAGE_SOURCE]}-{config[const.LANGUAGE_TARGET]}',
                                   split=const.TRAIN)
        tokenizer_source = cls._get_or_build_tokenizer(config, 
                                                       raw_dataset, 
                                                       config[const.LANGUAGE_SOURCE])
        tokenizer_target = cls._get_or_build_tokenizer(config, 
                                                       raw_dataset,
                                                       config[const.LANGUAGE_TARGET])
        dataset_size = len(raw_dataset)
        train_dataset_size = int(0.9*dataset_size)
        val_dataset_size = dataset_size - train_dataset_size

        train_dataset_raw, val_dataset_raw = random_split(raw_dataset,
                                                          [train_dataset_size, val_dataset_size])
        
        train_dataset = BilingualDataset(train_dataset_raw, 
                                         tokenizer_source,
                                         tokenizer_target,
                                         config[const.LANGUAGE_SOURCE],
                                         config[const.LANGUAGE_TARGET],
                                         config[const.SEQUENCE_LENGTH])
        validation_dataset = BilingualDataset(val_dataset_raw, 
                                              tokenizer_source,
                                              tokenizer_target,
                                              config[const.LANGUAGE_SOURCE],
                                              config[const.LANGUAGE_TARGET],
                                              config[const.SEQUENCE_LENGTH])
        cls._log_max_token_lengths(raw_dataset, tokenizer_source, tokenizer_target, config)

        train_dataloader = DataLoader(train_dataset, config[const.BATCH_SIZE], shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

        return train_dataloader, validation_dataloader, tokenizer_source, tokenizer_target


    @staticmethod
    def _log_max_token_lengths(dataset, tokenizer_source, tokenizer_target, config):
        max_len_source = 0
        max_len_target = 0
        for item in dataset:
            src_ids = tokenizer_source.encode(item[const.TRANSLATION][config[const.LANGUAGE_SOURCE]]).ids
            target_ids = tokenizer_target.encode(item[const.TRANSLATION][config[const.LANGUAGE_TARGET]]).ids
            max_len_source = max(max_len_source, len(src_ids))
            max_len_target = max(max_len_target, len(target_ids))
        print(f'Max length of source: {max_len_source}')
        print(f'Max length of target: {max_len_target}')

    @classmethod
    def _get_or_build_tokenizer(cls, config: dict, dataset, language) -> Tokenizer:
        tokenizer_path = Path(config[const.TOKENIZER_FILE].format(language))
        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token=const.TOKEN_UNKNOWN))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=[const.TOKEN_UNKNOWN, 
                                                       const.TOKEN_PADDING, 
                                                       const.TOKEN_START_OF_SENTENCE,
                                                       const.TOKEN_END_OF_SENTENCE], 
                                       min_frequency=2)
            tokenizer.train_from_iterator(cls._get_all_sentences(dataset, language),
                                          trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer        
    
    @staticmethod
    def _get_all_sentences(dataset, language):
        for item in dataset:
            yield item[const.TRANSLATION][language]

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = Config.get_config()
    Trainer.train_model(config)