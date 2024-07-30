import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import save_pfm
import ast
from new_utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONHASHSEED'] = '0'
seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Predict depth')
parser.add_argument('--dataset', default='data_eval_transform_padding', help='select dataset')
parser.add_argument('--testpath', default='/home/sby/data/MVS_Data/tankandtemples/intermediate/')
parser.add_argument('--testlist', default='lists/tp_list.txt', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--max_h', type=int, default=1056, help='Maximum image height when training')
parser.add_argument('--max_w', type=int, default=2048, help='Maximum image width when training.')

parser.add_argument('--view_num', type=int, default=7, help='training view num setting')
parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=True)

parser.add_argument('--base_image_size', type=int, default=8, help='crop base')
parser.add_argument('--numdepth', type=int, default=96, help='the number of depth values')

#parser.add_argument('--loadckpt', default='./checkpoints/0426-1536/model_000016.ckpt', help='load a specific checkpoint')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs_tnt', help='output dir')
parser.add_argument('--isTest', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=True)
parser.add_argument('--forTNT', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=True)
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')

# parse arguments and check
args = parser.parse_args()

# make save_dir
model_name = str.split(args.loadckpt, '/')[-2] + '_' + str.split(args.loadckpt, '/')[-1]
save_dir = os.path.join(args.outdir, model_name)
if not os.path.exists(save_dir):
    print('save dir', save_dir)
    os.makedirs(save_dir)

# create logger
logger = setup_logger("acinr-mvsnet", save_dir, prefix="test")
logger.info("################################  args  ################################")
for k, v in args.__dict__.items():
    logger.info("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
logger.info("########################################################################")

# run MVS model to save depth maps and confidence maps
def save_depth():
    logger = logging.getLogger("acinr-mvsnet.test")
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", 7, args.numdepth, args.inverse_depth,
                    max_h=args.max_h, max_w=args.max_w, base_image_size=args.base_image_size)

    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=16, drop_last=False)

    model = ACINRMVSNet(max_h=args.max_h, max_w=args.max_w, forTNT=args.forTNT, isTest=args.isTest)
    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint file specified by args.loadckpt
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()

    meters = MetricLogger(delimiter="  ")
    end = time.time()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            data_time = time.time() - end
            logger.info('process {}'.format(sample['filename']))
            sample_cuda = tocuda(sample)
            print('input img shape: ', sample_cuda["imgs"].shape)
            time_s = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], \
                            sample_cuda["extrinsics_list"], sample_cuda["intrinsics_list"], \
                            sample_cuda["image"], sample_cuda["hr_coord"], sample_cuda["lr_image"], sample_cuda["depth_bound"])
            model_time = time.time() - time_s
            batch_time = time.time() - end
            end = time.time()
            meters.update(batch_time=batch_time, data_time=data_time, model_time=model_time)

            outputs = tensor2numpy(outputs)
            del sample_cuda
            logger.info('Iter {}/{}, {}'.format(batch_idx + 1, len(TestImgLoader), str(meters)))
            filenames = sample["filename"]
            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence in zip(filenames, outputs["depth"],outputs["photometric_confidence"]):
                depth_filename = os.path.join(save_dir, filename.format('depth_est_{}'.format(0), '.pfm'))
                confidence_filename = os.path.join(save_dir, filename.format('confidence_{}'.format(0), '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                save_pfm(depth_filename, depth_est.squeeze())
                save_pfm(confidence_filename, photometric_confidence.squeeze())

        logger.info("Test {}".format(meters.summary_str))

if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    print('save depth *******************\n')
    save_depth()