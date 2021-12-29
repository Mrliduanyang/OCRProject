from pathlib import Path
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description='A tool for proportionally randomizing dataset to produce file lists.')
    parser.add_argument('dataset_root', help='the dataset root path', type=str, default='data/ocr')
    parser.add_argument(
        '--split', help='', nargs=3, type=float, default=[0.7, 0.3, 0])

    return parser.parse_args()


def generate_list(args):
    dataset_root = Path(args.dataset_root)
    if sum(args.split) != 1.0:
        raise ValueError("划分比例之和必须为1")

    file_list = dataset_root.joinpath('rec_gt.txt')
    train_file = dataset_root.joinpath('train_rec_gt.txt')
    val_file = dataset_root.joinpath('val_rec_gt.txt')
    # 将rec_gt.txt 随机分成三种，训练，验证，测试
    with open(file_list, 'r')as f:
        all_gt = f.readlines()
        random.shuffle(all_gt)
        start, end = 0, int(len(all_gt) * args.split[0])

        with open(train_file, 'w')as wf1:
            for item in all_gt[start: end]:
                wf1.write(item)
            start = end
            end = len(all_gt)

        with open(val_file, 'w')as wf2:
            for item in all_gt[start: end]:
                wf2.write(item)

if __name__ == '__main__':
    args=parse_args()
    generate_list(args)
