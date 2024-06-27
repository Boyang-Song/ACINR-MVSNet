import argparse
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
from new_utils import *
import ast
from typing import List
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONHASHSEED'] = '0'
seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(seed)

def create_stage_images(image: torch.Tensor) -> List[torch.Tensor]:
    return [
        F.interpolate(image.unsqueeze(1), (image.shape[1]*4,image.shape[2]*4), mode="nearest").squeeze(1),
        F.interpolate(image.unsqueeze(1), (image.shape[1]*2,image.shape[2]*2), mode="nearest").squeeze(1),
        F.interpolate(image.unsqueeze(1), (image.shape[1],image.shape[2]), mode="nearest").squeeze(1),
        F.interpolate(image.unsqueeze(1), (image.shape[1]//2,image.shape[2]//2), mode="nearest").squeeze(1)
    ]

def group_weight(module, weight_decay):
    group_decay = []
    group_no_decay = []
    keywords = [".bn."]

    for m in list(module.named_parameters()):
        exclude = False
        for k in keywords:
            if k in m[0]:
                print("Weight decay exclude: "+m[0])
                group_no_decay.append(m[1])
                exclude = True
                break
        if not exclude:
            print("Weight decay include: " + m[0])
            group_decay.append(m[1])

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay, weight_decay=weight_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


parser = argparse.ArgumentParser(description='PyTorch Codebase for ACINR-MVSNet')
parser.add_argument('--mode', default='train', help='train, val or test')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', default="/home/sby/data/MVS_Data/dtu_training/",help='train datapath')
parser.add_argument('--trainlist', default='lists/dtu/train.txt', help='train list')
parser.add_argument('--vallist', default='lists/dtu/val.txt', help='val list')
parser.add_argument('--testlist', default='lists/dtu/test.txt', help='test list')

parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')

parser.add_argument('--view_num', type=int, default=3, help='training view num setting')
parser.add_argument('--light_idx', type=int, default=-1, help='train 1-7(-1)/test 3')
parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=True)
parser.add_argument('--numdepth', type=int, default=48, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=4.24, help='the number of depth values')

parser.add_argument('--max_h', type=int, default=512, help='Maximum image height when training')
parser.add_argument('--max_w', type=int, default=640, help='Maximum image width when training.')

parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')

parser.add_argument('--save_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--save_dir', default=None, help='the directory to save checkpoints/logs')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--forTNT', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')


# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None

# create logger
if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)

logger = setup_logger("acinr-mvsnet", args.logdir, prefix="train")
logger.info("################################  args  ################################")
for k, v in args.__dict__.items():
    logger.info("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
logger.info("########################################################################")

logger.info("creating new summary file")
tensorboard_logger = SummaryWriter(args.logdir)
logger = logging.getLogger("acinr-mvsnet.train")

# save depth
SAVE_DEPTH = args.save_depth
if SAVE_DEPTH:
    if args.save_dir is None:
        sub_dir, ckpt_name = os.path.split(args.loadckpt)
        index = ckpt_name[6:-5]
        save_dir = os.path.join(sub_dir, index)
    else:
        save_dir = args.save_dir
    print(os.path.exists(save_dir), ' exists', save_dir)
    if not os.path.exists(save_dir):
        print('save dir', save_dir)
        os.makedirs(save_dir)

# dataset/model/loss/optimizer
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.view_num, args.numdepth, args.interval_scale, \
                           args.inverse_depth, light_idx=args.light_idx, max_h=args.max_h, max_w=args.max_w)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, drop_last=True, worker_init_fn=seed_worker, generator=g)

logger.info('model: ACINR-MVSNet')
model = ACINRMVSNet(max_h=args.max_h, max_w=args.max_w, forTNT=args.forTNT)
model = model.cuda()
model = nn.parallel.DataParallel(model)

logger.info('loss:mae-loss')
model_loss = MVS_MAE_Loss()
if args.forTNT:
    logger.info('optimizer:Adam')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
else:
    logger.info('optimizer:RMSprop')
    optimizer = optim.RMSprop(group_weight(model, weight_decay=args.wd), lr=args.lr, alpha=0.9)

# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    logger.info("resuming from:{}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    logger.info(optimizer)

    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
logger.info("start at epoch {}".format(start_epoch))
logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# main function
def train():
    logger.info('Run train()')
    if args.forTNT:
        milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
        lr_gamma = 1 / float(args.lrepochs.split(':')[1])
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                        last_epoch=len(TrainImgLoader) * start_epoch - 1)
        for epoch_idx in range(start_epoch, args.epochs):
            logger.info('Epoch {}/{}:'.format(epoch_idx + 1, args.epochs))
            logger.info('Start Training')
            meters = MetricLogger(delimiter="  ")

            for batch_idx, sample in enumerate(TrainImgLoader):
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                do_summary_image = global_step % (50 * args.summary_freq) == 0
                loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
                meters.update(loss=loss)
                lr_scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                if do_summary:
                    save_scalars(tensorboard_logger, 'train', scalar_outputs, global_step)
                    tensorboard_logger.add_scalar('train/lr', lr, global_step)
                if do_summary_image:
                    save_images(tensorboard_logger, 'train', image_outputs, global_step)
                del scalar_outputs, image_outputs
                if batch_idx % 10 == 0:
                    logger.info(
                        'Epoch {}/{}, Iter {}/{}, {}, time = {:.3f}, lr = {:.2e}'.format( \
                            epoch_idx + 1, args.epochs, batch_idx + 1, len(TrainImgLoader), str(meters),
                            time.time() - start_time, lr))

            # checkpoint
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx + 1))
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9, last_epoch=start_epoch - 1)
        for epoch_idx in range(start_epoch, args.epochs):
            logger.info('Epoch {}/{}:'.format(epoch_idx + 1, args.epochs))
            logger.info('Start Training')
            meters = MetricLogger(delimiter="  ")

            for batch_idx, sample in enumerate(TrainImgLoader):
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                do_summary_image = global_step % (50 * args.summary_freq) == 0
                loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
                meters.update(loss=loss)
                lr = optimizer.param_groups[0]["lr"]
                if do_summary:
                    save_scalars(tensorboard_logger, 'train', scalar_outputs, global_step)
                    tensorboard_logger.add_scalar('train/lr', lr, global_step)
                if do_summary_image:
                    save_images(tensorboard_logger, 'train', image_outputs, global_step)
                del scalar_outputs, image_outputs
                if batch_idx % 10 == 0:
                    logger.info(
                        'Epoch {}/{}, Iter {}/{}, {}, time = {:.3f}, lr = {:.2e}'.format( \
                            epoch_idx + 1, args.epochs, batch_idx + 1, len(TrainImgLoader), str(meters),
                            time.time() - start_time, lr))

            lr_scheduler.step()
            # checkpoint
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx + 1))


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = create_stage_images(sample_cuda["depth"])
    mask = create_stage_images(sample_cuda["mask"])
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], \
                    sample_cuda["extrinsics_list"], sample_cuda["intrinsics_list"], \
                    sample_cuda["image"], sample_cuda["hr_coord"], sample_cuda["lr_image"], sample_cuda["depth_bound"])

    depth_interval = sample_cuda["depth_params"][:,1]
    loss_dict = model_loss(outputs["coarse_depth"], outputs["flow1"], outputs["flow2"], \
                           sample_cuda["mask_depth"], depth_interval)

    loss = sum(loss_dict.values())
    loss.backward()
    optimizer.step()

    depth_flow2 = outputs["depth"]
    depth_flow1 = outputs["flow1"].squeeze()
    depth_init = outputs["coarse_depth"].squeeze()
    scalar_outputs = {"loss": loss}
    if args.forTNT:
        image_outputs = {
            "depth_gt": depth_gt[1] * mask[1],
            "depth_flow2": depth_flow2 * mask[1],
            "depth_flow1": depth_flow1 * mask[1],
            "depth_init": depth_init * mask[2],
            "ref_img": sample["imgs"][0]
        }
        if detailed_summary:
            image_outputs["errormap_depth_flow2"] = (depth_flow2 - depth_gt[1]).abs() * mask[1]
            image_outputs["errormap_depth_flow1"] = (depth_flow1 - depth_gt[1]).abs() * mask[1]
            image_outputs["errormap_depth_init"] = (depth_init - depth_gt[2]).abs() * mask[2]

            scalar_outputs["abs_depth_error_depth_init"] = absolute_depth_error_metrics(
                depth_init, depth_gt[2], mask[2] > 0.5)
            scalar_outputs["abs_depth_error_depth_flow1"] = absolute_depth_error_metrics(
                depth_flow1, depth_gt[1], mask[1] > 0.5)
            scalar_outputs["abs_depth_error_depth_flow2"] = absolute_depth_error_metrics(
                depth_flow2, depth_gt[1], mask[1] > 0.5)
            # threshold = 1mm
            scalar_outputs["thres1mm_error"] = threshold_metrics(depth_flow2, depth_gt[1], mask[1] > 0.5, 1.0)
            # threshold = 2mm
            scalar_outputs["thres2mm_error"] = threshold_metrics(depth_flow2, depth_gt[1], mask[1] > 0.5, 2.0)
            # threshold = 4mm
            scalar_outputs["thres4mm_error"] = threshold_metrics(depth_flow2, depth_gt[1], mask[1] > 0.5, 4.0)
            # threshold = 8mm
            scalar_outputs["thres8mm_error"] = threshold_metrics(depth_flow2, depth_gt[1], mask[1] > 0.5, 8.0)
    else:
        image_outputs = {
            "depth_gt": depth_gt[0] * mask[0],
            "depth_flow2": depth_flow2 * mask[0],
            "depth_flow1": depth_flow1 * mask[1],
            "depth_init": depth_init * mask[2],
            "ref_img": sample["imgs"][0]
        }
        if detailed_summary:
            image_outputs["errormap_depth_flow2"] = (depth_flow2 - depth_gt[0]).abs() * mask[0]
            image_outputs["errormap_depth_flow1"] = (depth_flow1 - depth_gt[1]).abs() * mask[1]
            image_outputs["errormap_depth_init"] = (depth_init - depth_gt[2]).abs() * mask[2]

            scalar_outputs["abs_depth_error_depth_init"] = absolute_depth_error_metrics(
                depth_init, depth_gt[2], mask[2] > 0.5)
            scalar_outputs["abs_depth_error_depth_flow1"] = absolute_depth_error_metrics(
                depth_flow1, depth_gt[1], mask[1] > 0.5)
            scalar_outputs["abs_depth_error_depth_flow2"] = absolute_depth_error_metrics(
                depth_flow2, depth_gt[0], mask[0] > 0.5)
            # threshold = 1mm
            scalar_outputs["thres1mm_error"] = threshold_metrics(depth_flow2, depth_gt[0], mask[0] > 0.5, 1.0)
            # threshold = 2mm
            scalar_outputs["thres2mm_error"] = threshold_metrics(depth_flow2, depth_gt[0], mask[0] > 0.5, 2.0)
            # threshold = 4mm
            scalar_outputs["thres4mm_error"] = threshold_metrics(depth_flow2, depth_gt[0], mask[0] > 0.5, 4.0)
            # threshold = 8mm
            scalar_outputs["thres8mm_error"] = threshold_metrics(depth_flow2, depth_gt[0], mask[0] > 0.5, 8.0)


    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()