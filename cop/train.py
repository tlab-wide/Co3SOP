import argparse
import yaml
import time
import os
import torch
import importlib
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.logger import logger
from utils.data import Ego, V2V
from utils.metrics import Metrics
from torch.profiler import profile, record_function, ProfilerActivity

tensor_writer = None

def train(model, dataset, metrics: Metrics, configs: dict):
    global tensor_writer

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    epochs = configs["model_params"]["epochs"]

    for epoch in range(epochs):
        logger.info(f'=> =============== Epoch [{epoch+1}/{epochs}] ===============')
        metrics.reset()
        for t, data in enumerate(dataset):
            # logger.info(f'=> =============== Epoch [{epoch}/{epochs}] Iteration: {t}/{len(dataset)}===============')
            # with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
            for k, v in data.items():
                data[k] = v.to(device=device, dtype=dtype)
                # data_copy[k] = data[k].clone().detach()

            # with record_function("forward"):
            output = model(data)

            # with record_function("backward"):
            loss = model.optimize_step(output, data)

            with torch.no_grad():

                # with record_function("metrics"):
                metrics.add_batch(output, model.get_target(data), loss)

                if (t + 1) % int(configs['period']['logger']) == 0:
                    loss_print = '=> Epoch [{}/{}], Iteration [{}/{}], Learn Rate: {}, Train Losses: {}'\
                    .format(epoch+1, epochs, t+1, len(dataset), model.scheduler.get_lr()[0], loss)

                    logger.info(loss_print[:-3])
                    
                # print(metrics.get_stats())
                # return

            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        tensor_writer.add_scalar('train_loss_epoch',metrics.total_loss/metrics.iteration,epoch)
        tensor_writer.add_scalar('learning rate', model.scheduler.get_lr()[0], epoch)


        # for scale in metrics.evaluator.keys():
        #     tensor_writer.add_scalar('train_performance/{}/mIoU'.format(scale), metrics.get_semantics_mIoU(scale).item(), epoch-1)
        #     tensor_writer.add_scalar('train_performance/{}/IoU'.format(scale), metrics.get_occupancy_IoU(scale).item(), epoch-1)


        logger.info('=> [Epoch {} - Total Train Loss = {}]'.format(epoch+1, metrics.total_loss/metrics.iteration))
        stats = metrics.get_stats()

        # for scale in metrics.evaluator.keys():
        #     loss_scale = metrics.losses_track.train_losses['semantic_{}'.format(scale)].item()/metrics.losses_track.train_iteration_counts
        logger.info('=> [Epoch {}/{} : Loss = {} - mIoU = {} - Completion = {}]'
                    .format(epoch+1, epochs,
                            metrics.total_loss/metrics.iteration,
                            stats["mIoU"],
                            stats["completion"]))

        for i in range(metrics.n_classes):
            class_name  = configs['labels'][i]
        # class_score = metrics.evaluator['1'].getIoU()[1][i]
            logger.info('    => IoU {}: {:.6f}'.format(class_name, stats["IoU"][i]))

        # # Reset evaluator and losses for next epoch...
        
        if configs['model_params']['scheduler_frequency'] == 'epoch':
            model.scheduler.step()
        if epoch + 1 % configs['period']['checkpoint'] == 0:
            checkpoint_path = os.path.join(configs['output']['root'], 'ckpt', str(epoch+1).zfill(2), 'weights_epoch_{}.pth'.format(str(epoch+1).zfill(3)))
            if not os.path.exists(os.path.dirname(checkpoint_path)):
                os.makedirs(os.path.dirname(checkpoint_path))
            torch.save({
                'startEpoch': epoch+2,  # To start on next epoch when loading the dict...
                'model': model.state_dict(),
                'optimizer': model.optimizer.state_dict(),
                'scheduler': model.scheduler.state_dict(),
                'config_dict': configs,
                'metrics': {}
            }, checkpoint_path)

    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', type=str, help='config path', default=
                           "configs/LMSCNet.yaml")
    args = argparser.parse_args()
    configs = {}
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    with open("configs/labels.yaml", "r") as f:
        configs.update(yaml.load(f, Loader=yaml.FullLoader))
    
    model = importlib.import_module(f"models.{configs['model_params']['name']}")
    if configs["model_params"]["name"] != "COP-3D":
        dataset = torch.utils.data.DataLoader(
            dataset=Ego(configs),
            batch_size=configs["dataset"]["batch_size"],
            # collate_fn=model.data_adaption,
            shuffle=bool(configs["dataset"]["shuffle"]),
            num_workers=configs["dataset"]["num_workers"],
            pin_memory=True,
        )
    else:
        dataset = torch.utils.data.DataLoader(
            dataset=V2V(configs),
            batch_size=configs["dataset"]["batch_size"],
            # collate_fn=model.data_adaption,
            shuffle=configs["dataset"]["shuffle"],
            num_workers=configs["dataset"]["num_workers"]
        )

    model = getattr(model, configs["model_params"]["name"])(configs)
    model.weights_init()

    metrics = Metrics(len(configs["labels"]))

    tensor_writer = SummaryWriter(log_dir=os.path.join(configs['output']['root'], 'metrics'))

    train(model, dataset, metrics, configs)