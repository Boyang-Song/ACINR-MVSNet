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


parser = argparse.ArgumentParser(description='Validate')
parser.add_argument('--dataset', default='dtu_yao_blend_caspl', help='select dataset')
parser.add_argument('--testpath', default='/home/sby/data/MVS_Data/dataset_low_res/')
parser.add_argument('--testlist', default='/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--max_h', type=int, default=576, help='Maximum image height when training')
parser.add_argument('--max_w', type=int, default=768, help='Maximum image width when training.')
parser.add_argument('--view_num', type=int, default=3, help='view num setting,test5/val3')
parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--numdepth', type=int, default=48, help='the number of depth values')

# parser.add_argument('--loadckpt', default='./checkpoints/pretrained/model_0301.ckpt', help='load a specific checkpoint')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./val_blend', help='output dir')
parser.add_argument('--forTNT', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

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
def validate():
    logger = logging.getLogger("acinr-mvsnet.test")
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.view_num, args.numdepth, args.inverse_depth,
                              max_h=args.max_h, max_w=args.max_w)

    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=16, drop_last=False)

    model = ACINRMVSNet(max_h=args.max_h, max_w=args.max_w, forTNT=args.forTNT)
    model = nn.DataParallel(model)
    model.cuda()
    # load checkpoint file specified by args.loadckpt
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()

    logger.info('loss:mae-loss')
    loss_fn = MVS_MAE_Loss()
    metric_fn = BlendMetric()

    meters = MetricLogger(delimiter="  ")
    end = time.time()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            data_time = time.time() - end
            logger.info('process {}'.format(sample['name']))
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

            depth_interval = sample_cuda["depth_params"][:, 1]
            loss_dict = loss_fn(outputs["coarse_depth"], outputs["flow1"], outputs["flow2"], \
                                   sample_cuda["mask_depth"], depth_interval)
            metric_dict = metric_fn(outputs["coarse_depth"], outputs["flow1"], outputs["flow2"], \
                                   sample_cuda["mask_depth"], depth_interval)
            losses = sum(loss_dict.values())
            meters.update(loss=losses, **loss_dict, **metric_dict)

            del sample_cuda
            logger.info('Iter {}/{}, {}'.format(batch_idx + 1, len(TestImgLoader), str(meters)))


        logger.info("Test {}".format(meters.summary_str))

if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    print('validate *******************\n')
    validate()
