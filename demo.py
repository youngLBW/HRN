import os
from models.hrn import Reconstructor
import cv2
from tqdm import tqdm
import argparse


def run_hrn(args):
    params = [
        '--checkpoints_dir', args.checkpoints_dir,
        '--name', args.name,
        '--epoch', args.epoch,
    ]

    reconstructor = Reconstructor(params)

    names = sorted([name for name in os.listdir(args.input_root) if '.jpg' in name or '.png' in name or '.jpeg' in name or '.PNG' in name or '.JPG' in name or '.JPEG' in name])

    print('predict', args.input_root)

    for ind, name in enumerate(tqdm(names)):
        save_name = os.path.splitext(name)[0]
        out_dir = os.path.join(args.output_root, save_name)
        os.makedirs(out_dir, exist_ok=True)
        img = cv2.imread(os.path.join(args.input_root, name))
        output = reconstructor.predict(img, visualize=True, save_name=save_name, out_dir=out_dir)

    print('results are saved to:', args.output_root)


def run_mvhrn(args):
    params = [
        '--checkpoints_dir', args.checkpoints_dir,
        '--name', args.name,
        '--epoch', args.epoch,
    ]

    reconstructor = Reconstructor(params)

    names = sorted([name for name in os.listdir(args.input_root) if
                    '.jpg' in name or '.png' in name or '.jpeg' in name or '.PNG' in name or '.JPG' in name or '.JPEG' in name])
    os.makedirs(args.output_root, exist_ok=True)

    print('predict', args.input_root)

    out_dir = args.output_root
    os.makedirs(out_dir, exist_ok=True)
    img_list = []
    for ind, name in enumerate(names):
        img = cv2.imread(os.path.join(args.input_root, name))
        img_list.append(img)
        # output = reconstructor.predict_base(img, save_name=save_name, out_dir=out_dir)
    output = reconstructor.predict_multi_view(img_list, visualize=True, out_dir=out_dir)

    print('results are saved to:', args.output_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--checkpoints_dir', type=str, default='assets/pretrained_models', help='models are saved here')
    parser.add_argument('--name', type=str, default='hrn_v1.1',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--epoch', type=str, default='10', help='which epoch to load? set to latest to use latest cached model')

    parser.add_argument('--input_type', type=str, default='single_view',  # or 'multi_view'
                        help='reconstruct from single-view or multi-view')
    parser.add_argument('--input_root', type=str, default='./assets/examples/single_view_image',
                        help='directory of input images')
    parser.add_argument('--output_root', type=str, default='./assets/examples/single_view_image_results',
                        help='directory for saving results')

    args = parser.parse_args()

    if args.input_type == 'multi_view':
        run_mvhrn(args)
    else:
        run_hrn(args)