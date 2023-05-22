import tqdm
import torch
import diffusers

def batch_diffusion(hparams, 
                    batch_data:torch.Tensor, 
                    ddpm_noise_sched:diffusers.DDIMScheduler, 
                    ddpm_model:diffusers.UNet2DModel):
    
    batch_noise = torch.randn_like(batch_data, dtype=torch.float32, device=batch_data.device)
    batch_noise_step = torch.randint(0, hparams.diffuse_step, size=[len(batch_data),], device=batch_data.device).long()
    batch_noise_data = ddpm_noise_sched.add_noise(batch_data, noise=batch_noise, timesteps=batch_noise_step)
    batch_out = ddpm_model.forward(sample=batch_noise_data, timestep=batch_noise_step)
    return batch_noise, batch_out.sample

@torch.no_grad()
def batch_inverse(hparams, 
                  ddpm_noise_sched:diffusers.DDIMScheduler, 
                  ddpm_model:diffusers.UNet2DModel):
    batch_x = torch.randn([hparams.N_eval,3,hparams.img_size,hparams.img_size], dtype=torch.float32, device=hparams.device)
    pbar_inverse = tqdm.tqdm(reversed(range(1, hparams.diffuse_step)), position=0, desc="reversing", leave=False)
    for stepi in pbar_inverse:
        batch_ti = (torch.ones(hparams.N_eval, device=hparams.device) * stepi).long()
        batch_out = ddpm_model.forward(batch_x, batch_ti)
        batch_noise_pred = batch_out.sample
        if stepi > 1:
            noise = torch.randn_like(batch_x, device=batch_x.device).float()
        else:
            noise = torch.zeros_like(batch_x, device=batch_x.device).float()
        alphai = ddpm_noise_sched.alphas[batch_ti][..., None, None, None].to(hparams.device)
        alphas_cumprodi = ddpm_noise_sched.alphas_cumprod[batch_ti][..., None, None, None].to(hparams.device)
        betai = ddpm_noise_sched.betas[batch_ti][..., None, None, None].to(hparams.device)
        batch_x = 1 / torch.sqrt(alphai) * (batch_x - ((1 - alphai) / (torch.sqrt(1 - alphas_cumprodi))) * batch_noise_pred) + torch.sqrt(betai) * noise
    batch_x = (batch_x.clamp(-1., 1.) + 1.) / 2.
    batch_x = (batch_x * 255).type(torch.uint8)
    return batch_x