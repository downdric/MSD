import os
import shutil


def get_label(text_path):
    res = {}
    for line in open(text_path, 'r').readlines():
        content = eval(line)
        skip_words = ['exgag', 'sarcasm', 'sarcastic', '<url>', 'reposting', 'joke', 'humor', 'humour', 'jokes', 'irony', 'ironic']
        flag = False
        for skip_word in skip_words:
            if skip_word in content[1]: flag = True
        if flag: continue
        res[content[0]] = content[2]
    return res


def mv_img(labels, root_dir, target_dir):
    anno = ['non-sarcasm', 'sarcasm']
    for cur_anno in anno:
        cur_dir = os.path.join(target_dir, cur_anno)
        if not os.path.exists(cur_dir): os.makedirs(cur_dir)
    
    for cur_key in labels.keys():
        cur_img = cur_key + '.jpg'
        target_path = os.path.join(target_dir, anno[labels[cur_key]], cur_img)
        origin_path = os.path.join(root_dir, cur_img)
        shutil.copy(origin_path, target_path)

