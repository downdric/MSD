import torch
from transformers import BertTokenizer
from torchtext import data, datasets
# from torchtext.data import Example
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import tqdm
from torchcrf import CRF
import numpy as np
from transformers import AutoTokenizer
import torchtext.vocab as vocab


def initialize_tokenizer(args):
    if args.text_model == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    elif args.text_model == 'large':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        max_input_length = tokenizer.max_model_input_sizes['bert-large-uncased']
    else:
        print('Only support base and large')
        exit(0)
    print(len(tokenizer.vocab))

    print(max_input_length)

    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    print(init_token, eos_token, pad_token, unk_token)

    return tokenizer


def generate_text_loader(args, device, tokenizer, max_input_length, init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx):
    def tokenize_and_cut(sentence):
        tokens = tokenizer.tokenize(sentence) 
        tokens = tokens[:max_input_length-2]
        return tokens

    TEXT = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = tokenize_and_cut,
                    preprocessing = tokenizer.convert_tokens_to_ids,
                    init_token = init_token_idx,
                    eos_token = eos_token_idx,
                    pad_token = pad_token_idx,
                    unk_token = unk_token_idx)

    LABEL = data.LabelField(dtype = torch.float)

    train_fields = [('text', TEXT), ('label', LABEL)]
    test_fields = [('text', TEXT), ('label', LABEL)]
    train_examples = []
    test_examples = []

    skip_words = ['exgag', 'sarcasm', 'sarcastic', '<url>', 'reposting', 'joke', 'humor', 'humour', 'jokes', 'irony', 'ironic']

    for line in tqdm.tqdm(open(args.train_text_path, 'r').readlines()):
        content = eval(line)
        flag = False
        for skip_word in skip_words:
            if skip_word in content[1]: flag = True
        if flag: continue
        text, label = content[1], content[2]
        # train_examples.append(Example.fromlist([text, label], train_fields))
    
    for line in tqdm.tqdm(open(args.test_text_path, 'r').readlines()):
        content = eval(line)
        # flag = False
        # for skip_word in skip_words:
        #     if skip_word in content[1]: flag = True
        # if flag: continue
        text, label = content[1], content[2]
        # test_examples.append(Example.fromlist([text, label], test_fields))

    train_set = datasets(train_examples, train_fields)
    test_set = datasets(test_examples, test_fields)

    LABEL.build_vocab(train_set)

    train_iterator, test_iterator = data.BucketIterator.splits((train_set, test_set), batch_size=args.batch_size, device=device, sort=False)

    return train_iterator, test_iterator


class BERT_MODEL(nn.Module):
    def __init__(self, bert, output_dim, alg, embedding=False):
        super().__init__()
        self.bert = bert
        self.embedding = embedding
        if alg == 'base': self.fc = nn.Linear(768, output_dim)
        else: self.fc = nn.Linear(1024, output_dim)
        
    def forward(self, text):
        output = self.bert(text)
        res = {'embeddings': output['last_hidden_state'][:, 1:, :], 'cls': output['pooler_output']}
        return res


class LSTM_MODEL(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(vocab.GloVe(name='42B', dim=300).vectors)

        self.biLSTM = nn.LSTM(input_size=300, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, 768, bias=True)
        # self.biLSTM = nn.LSTM(input_size=300, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self.fc = nn.Linear(hidden_size, 768, bias=True)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.biLSTM(embedded)
        output = self.fc(output)
        # output = F.relu(self.fc(output))

        return  {'embeddings': output}


def get_text_model(args, embedding=False):
    if args.text_backbone == 'bert':
        if args.text_model == 'base':
            bert = BertModel.from_pretrained('bert-base-uncased')
        elif args.text_model == 'large':
            bert = BertModel.from_pretrained('bert-large-uncased')
        else:
            print('error and tokenizer may have something wrong')
            exit(0)
        model = BERT_MODEL(bert, args.output_dim, args.text_model, embedding=embedding)
    else:
        model = LSTM_MODEL(hidden_size=256)

    return model


def get_text_configuration(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.text_lr, weight_decay=args.text_weight_decay)
    num_training_steps = int(args.train_set_len / args.batch_size * args.epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion = nn.BCEWithLogitsLoss()
    return optimizer, scheduler, criterion
