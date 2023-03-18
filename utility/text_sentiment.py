from senticnet.senticnet import SenticNet
import numpy as np
import torch


sn = SenticNet()
def get_word_level_sentiment(texts, tokenizer, device):
    res = []
    for text in texts:
        if tokenizer is not None: word_list = tokenizer.tokenize(text)
        else: word_list = text.split()
        text_res = []
        for word in word_list:
            try:
                word_polarity_value = float(sn.concept(word)['polarity_value'])
            except:
                word_polarity_value = float(0)
            text_res.append(word_polarity_value)
        res.append(torch.tensor(text_res).to(device))
    return res


def get_text_sentiment(texts, tokenizer, device):
    res = []
    for text in texts:
        if tokenizer is not None: word_list = tokenizer.tokenize(text)
        else: word_list = text.split()
        text_res, cnt = 0, 0
        for word in word_list:
            try:
                text_res += float(sn.concept(word)['polarity_value'])
                cnt += 1
            except:
                text_res += float(0)
        if cnt != 0: text_res = text_res / cnt
        else: cnt = 0
        res.append(text_res)
    res = torch.tensor(res).to(device)
    return res
