import os
from shutil import copyfile
import sys


def generate_multitask_dataset(main_task_data_path, sub_task_data_path, output_path):
    mainData = dict()
    row_order = list()
    with open(main_task_data_path, 'r') as mStream:
        for line in mStream:
            items = line.strip().split("\t")
            rid = items[2]
            mainData[rid] = [items[0], items[1]]
            row_order.append(rid)

    with open(sub_task_data_path, 'r') as sStream:
        for line in sStream:
            items = line.strip().split("\t")
            rid = items[2]
            assert rid in mainData, 'missing rid'
            mainData[rid] += [items[1]]

    with open(output_path, 'w') as ostream:
        for rid in row_order:
            assert len(mainData[rid]) == 3, 'malformed row'
            serial_row = '\t'.join(mainData[rid] + [rid])
            ostream.write(f'{serial_row}\n')


if __name__ == '__main__':
    subtask = sys.argv[1]
    os.makedirs(f"GoEmotions-pytorch-multitask/data/multitask-{subtask}", exist_ok=True)
    generate_multitask_dataset(
        "GoEmotions-pytorch-multitask/data/original/train.tsv",
        f"GoEmotions-pytorch-multitask/data/{subtask}/train.tsv",
        f"GoEmotions-pytorch-multitask/data/multitask-{subtask}/train.tsv")
    copyfile("GoEmotions-pytorch-multitask/data/original/dev.tsv",
             f"GoEmotions-pytorch-multitask/data/multitask-{subtask}/dev.tsv")
    copyfile("GoEmotions-pytorch-multitask/data/original/test.tsv",
             f"GoEmotions-pytorch-multitask/data/multitask-{subtask}/test.tsv")
    copyfile("GoEmotions-pytorch-multitask/data/original/labels.txt",
             f"GoEmotions-pytorch-multitask/data/multitask-{subtask}/labels.txt")
    copyfile(f"GoEmotions-pytorch-multitask/data/{subtask}/labels.txt",
             f"GoEmotions-pytorch-multitask/data/multitask-{subtask}/subtask-labels.txt")
