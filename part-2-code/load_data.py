import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

# Global tokenizer
TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")

PAD_ID = TOKENIZER.pad_token_id
EOS_ID = TOKENIZER.eos_token_id
BOS_TOKEN = "<extra_id_0>"          # any extra-id can be used as BOS
BOS_ID = TOKENIZER.convert_tokens_to_ids(BOS_TOKEN)



class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.data_folder = data_folder
        self.split = split  # "train", "dev", or "test"

        self.tokenizer = TOKENIZER
        self.pad_id = PAD_ID
        self.eos_id = EOS_ID
        self.bos_id = BOS_ID

        self.encoder_input_ids = []
        self.decoder_input_ids = []
        self.decoder_target_ids = []

        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)

        # For train/dev we also have SQL targets
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_lines = load_lines(sql_path)
            assert len(nl_lines) == len(sql_lines), f"Line count mismatch in {split} NL/SQL."

            for nl, sql in zip(nl_lines, sql_lines):
                # Encoder input: natural language question
                enc_ids = tokenizer.encode(nl, add_special_tokens=False)

                # Decoder target: SQL query
                tgt_ids = tokenizer.encode(sql, add_special_tokens=False)

                # Teacher forcing:
                #   decoder inputs:  [BOS, y0, y1, ..., y_{n-1}]
                #   decoder targets: [y0,  y1, ..., y_{n-1}, EOS]
                dec_in = [self.bos_id] + tgt_ids
                dec_tgt = tgt_ids + [self.eos_id]

                self.encoder_input_ids.append(enc_ids)
                self.decoder_input_ids.append(dec_in)
                self.decoder_target_ids.append(dec_tgt)

        else:
            # Test set: only NL inputs, no targets
            for nl in nl_lines:
                enc_ids = tokenizer.encode(nl, add_special_tokens=False)
                self.encoder_input_ids.append(enc_ids)
    
    def __len__(self):
        return len(self.encoder_input_ids)

    def __getitem__(self, idx):
        if self.split == "test":
            # Only encoder inputs for test
            return {
                "encoder_ids": self.encoder_input_ids[idx]
            }
        else:
            # Train/dev have encoder + decoder inputs/targets
            return {
                "encoder_ids": self.encoder_input_ids[idx],
                "decoder_inputs": self.decoder_input_ids[idx],
                "decoder_targets": self.decoder_target_ids[idx],
            }
        

def pad_sequences(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    padded = [s + [pad_id] * (max_len - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long)


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_seqs = [item["encoder_ids"] for item in batch]
    decoder_in_seqs = [item["decoder_inputs"] for item in batch]
    decoder_tgt_seqs = [item["decoder_targets"] for item in batch]

    # Pad encoder and decoder sequences
    encoder_ids = pad_sequences(encoder_seqs, PAD_ID)
    decoder_inputs = pad_sequences(decoder_in_seqs, PAD_ID)
    decoder_targets = pad_sequences(decoder_tgt_seqs, PAD_ID)

    # Encoder mask: 1 for non-padding, 0 for padding
    encoder_mask = (encoder_ids != PAD_ID).long()

    # Initial decoder inputs: BOS token, shape B x 1
    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full((batch_size, 1), BOS_ID, dtype=torch.long)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs




def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_seqs = [item["encoder_ids"] for item in batch]

    encoder_ids = pad_sequences(encoder_seqs, PAD_ID)
    encoder_mask = (encoder_ids != PAD_ID).long()

    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full((batch_size, 1), BOS_ID, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))

    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))

    test_x = load_lines(os.path.join(data_folder, "test.nl"))

    return train_x, train_y, dev_x, dev_y, test_x


def get_preprocessed_statistics(dset):
    """
    Compute the required statistics *after preprocessing*
    using the tokenized sequences inside T5Dataset.
    
    Required statistics:
        - Number of examples
        - Mean sentence length (NL)
        - Mean SQL query length
        - Vocabulary size (NL)
        - Vocabulary size (SQL)
    """

    # All splits (train/dev/test) have NL sequences
    nl_seqs = dset.encoder_input_ids
    nl_lengths = [len(seq) for seq in nl_seqs]
    nl_vocab = set([tid for seq in nl_seqs for tid in seq])

    stats = {
        "num_examples": len(nl_seqs),
        "mean_sentence_length": sum(nl_lengths) / len(nl_lengths),
        "vocab_size_nl": len(nl_vocab),
    }

    # SQL exists only for train/dev
    if dset.split != "test":
        sql_target_seqs = dset.decoder_target_ids   # SQL targets
        sql_lengths = [len(seq) for seq in sql_target_seqs]
        sql_vocab = set([tid for seq in sql_target_seqs for tid in seq])

        stats["mean_sql_length"] = sum(sql_lengths) / len(sql_lengths)
        stats["vocab_size_sql"] = len(sql_vocab)
    else:
        # No SQL for test set
        stats["mean_sql_length"] = None
        stats["vocab_size_sql"] = None

    return stats






#train_set = T5Dataset("data", "train")
#dev_set = T5Dataset("data", "dev")
#test_set = T5Dataset("data", "test")

#train_stats = get_preprocessed_statistics(train_set)
#dev_stats = get_preprocessed_statistics(dev_set)
#test_stats = get_preprocessed_statistics(test_set)

#print("TRAIN:", train_stats)
#print("DEV:", dev_stats)
#print("TEST:", test_stats)