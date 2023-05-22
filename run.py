import tqdm
import copy
import torch
from utils.configs import config_parser
from utils.common import create_usage
from utils.ddpm import batch_diffusion, batch_inverse
from utils.wandb_helper import LOG
from utils.ema import EMA
from dataset.data import POKEMON_DATASET

if __name__ == "__main__":
    # NOTE Step: Preparation
    # * train configures
    hparams = config_parser()

    # * train dataset and dataloader
    dataset = POKEMON_DATASET(hparams)
    dataloader = dataset.set_loader(bs=hparams.batch_size, is_shuffle=True)
    hparams.max_step = hparams.max_epoch * len(dataloader)

    # * ddpm model, ddpm noise, optim, optim_sched, loss_func
    ddpm_model, ddpm_noise_sched, ddpm_optim, ddpm_optim_sched, ddpm_loss_func = create_usage(hparams)

    # * ema for better performance
    ema = EMA(hparams.ema_beta)
    ema_model = copy.deepcopy(ddpm_model).eval().requires_grad_(False)
   
    logger = LOG()
    logger.log_in(configs=hparams, name=hparams.exp_name, project="DDPM")
    logger.log_metric_init()

    # NOTE Step: Train
    # * training loop
    cur_iter = 0
    pbar_epoch = tqdm.tqdm(range(hparams.max_epoch), desc="train ddpm", leave=True)
    for cur_ep in pbar_epoch:
        pbar_iter = tqdm.tqdm(dataloader, desc=f"train epoch {cur_ep}", leave=False)
        loss_ep = 0
        for batch_data in pbar_iter:
            batch_data = batch_data.to(hparams.device)
            batch_noise, batch_out_sample = batch_diffusion(hparams, batch_data, ddpm_noise_sched, ddpm_model)
            loss = ddpm_loss_func(batch_noise, batch_out_sample)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm_model.parameters(), 1.0)
            ddpm_optim.step()
            ddpm_optim_sched.step()
            ddpm_optim.zero_grad()

            ema.step_ema(ema_model, ddpm_model)
            loss_iter = loss.item()
            loss_ep += loss_iter

            pbar_iter.set_postfix_str(f"[TRAIN] Iter: {cur_iter} Loss: {loss_iter:.3f}", refresh=True)

        pbar_epoch.set_postfix_str(f"[TRAIN] EP: {cur_ep} Loss: {loss_ep:.3f}", refresh=True)
        logger.log_value(tag="train", name="ep_loss", value=loss_ep, step=cur_ep)

        # * test loop
        if (cur_ep+1)%hparams.eval_epoch == 0:
            images_pred = batch_inverse(hparams, ddpm_noise_sched, ddpm_model)
            images_pred = images_pred.cpu().detach().clone().permute([0,2,3,1]).numpy()
            logger.log_images(tag="test", name="ddpm_generation", values=images_pred, step=cur_ep, n_col=5)

    logger.log_out()