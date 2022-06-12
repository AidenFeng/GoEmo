import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_confusion_matrix(label_seq1, label_seq2, id2label1, id2label2):
    paired_labels = list()
    for l1 in label_seq1:
        for l2 in label_seq2:
            if len(l1) > 1 or len(l2) > 1:
                break
            paired_labels.append((l1[0], l2[0],))
    res = dict()
    print('size of label_seq1: ', len(label_seq1))
    print('size of label_seq2: ', len(label_seq2))
    print('size of paired_labels: ', len(paired_labels))
    for x, y in paired_labels:
        res[x, y] = res.get((x, y,), 0) + 1
    label_res = dict()
    for (x, y), v in res.items():
        label_res[id2label1[x], id2label2[y]] = v
    labels1 = sorted(id2label1.values())
    labels2 = sorted(id2label2.values())
    m, n = len(labels1), len(labels2)
    res_matrix = np.zeros((m, n,))
    for i, l1 in enumerate(labels1):
        for j, l2 in enumerate(labels2):
            res_matrix[i][j] = label_res.get((l1, l2,), 1)
    return res_matrix, labels1, labels2


def load_labels(in_path):
    seq = list()
    with open(in_path, 'r') as istream:
        for line in istream:
            lbls = line.strip().split('\t')[1].split(',')
            seq.append(lbls)
    return seq


def get_load_mapping(in_path):
    label_mapping = dict()
    with open(in_path, 'r') as istream:
        for i, line in enumerate(istream):
            label_mapping[str(i)] = line.strip()
    return label_mapping


def cal_nmi(confusion):
    c = confusion.sum()
    c_i_sum = np.sum(confusion, axis=1, keepdims=True)
    c_j_sum = np.sum(confusion, axis=0, keepdims=True)
    unit1 = np.log2(confusion) + np.log2(c) - np.log2(c_i_sum) - np.log2(c_j_sum)
    unit2 = confusion / c
    return ((unit1 * unit2).sum()) / (-(unit2 * np.log2(unit2)).sum())


def main():
    path = 'GoEmotions-pytorch-multitask/data/{}/{}'
    split = 'train'
    task = 'group'
    label_seq1 = load_labels(path.format('original', f'{split}.tsv'))
    label_seq2 = load_labels(path.format(task, f'{split}.tsv'))
    print(len(label_seq1), len(label_seq2))
    mapping1 = get_load_mapping(path.format('original', 'labels.txt'))
    mapping2 = get_load_mapping(path.format(task, 'labels.txt'))
    print(len(mapping1), len(mapping2))
    confusion, labels1, labels2 = get_confusion_matrix(label_seq1, label_seq2, mapping1, mapping2)
    print(task, split)
    # print(confusion)
    print(cal_nmi(confusion))

    sns.set_theme()
    ax = sns.heatmap(np.log(confusion))
    plt.savefig(f'GoEmotions-pytorch-multitask/data/{task}-{split}-confusion.pdf')
    np.savetxt(f"GoEmotions-pytorch-multitask/data/{task}-{split}-confusion.csv", confusion, delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')


if __name__ == '__main__':
    main()
