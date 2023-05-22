import torch
import diffusers

def create_usage(hparams):

    ddpm_model = diffusers.UNet2DModel(
        sample_size=hparams.img_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128,128,
                            256,256,
                            512,512),
        down_block_types=("DownBlock2D", "DownBlock2D",
                          "DownBlock2D", "DownBlock2D",
                          "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", 
                        "UpBlock2D", "UpBlock2D", 
                        "UpBlock2D", "UpBlock2D"),
    ).to(hparams.device)

    sched_config = {
        "num_train_timesteps": hparams.diffuse_step,
        "beta_end": 0.012,
        "beta_start": 0.00085,
        "beta_schedule": "scaled_linear",
        "clip_sample": False,
        "clip_sample_range": 1.0,
        "dynamic_thresholding_ratio": 0.995,
        "prediction_type": "epsilon",
        "sample_max_value": 1.0,
        "thresholding": False,
        "trained_betas": None,
        "variance_type": "fixed_small"
    }
    ddpm_noise_sched = diffusers.DDPMScheduler.from_config(sched_config)

    ddpm_optim = torch.optim.AdamW(ddpm_model.parameters(), 
                                   lr=hparams.lr_init,
                                   betas=(0.95,0.999),
                                   weight_decay=1e-6,
                                   eps=1e-8)

    ddpm_optim_sched = diffusers.get_cosine_schedule_with_warmup(ddpm_optim,
                                                                 num_warmup_steps=hparams.max_warmup_step,
                                                                 num_training_steps=hparams.max_step)
    
    ddpm_loss_func = torch.nn.MSELoss()
    return ddpm_model, ddpm_noise_sched, ddpm_optim, ddpm_optim_sched, ddpm_loss_func