import os
from shutil import copyfile


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
    os.makedirs("GoEmotions-pytorch-multitask/data/multitask", exist_ok=True)
    generate_multitask_dataset(
        "GoEmotions-pytorch-multitask/data/original/train.tsv",
        "GoEmotions-pytorch-multitask/data/group/train.tsv",
        "GoEmotions-pytorch-multitask/data/multitask/train.tsv")
    copyfile("GoEmotions-pytorch-multitask/data/original/dev.tsv",
             "GoEmotions-pytorch-multitask/data/multitask/dev.tsv")
    copyfile("GoEmotions-pytorch-multitask/data/original/test.tsv",
             "GoEmotions-pytorch-multitask/data/multitask/test.tsv")
    copyfile("GoEmotions-pytorch-multitask/data/original/labels.txt",
             "GoEmotions-pytorch-multitask/data/multitask/labels.txt")
    copyfile("GoEmotions-pytorch-multitask/data/group/labels.txt",
             "GoEmotions-pytorch-multitask/data/multitask/subtask-labels.txt")
