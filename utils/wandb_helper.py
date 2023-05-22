import os
import wandb
import termcolor
import numpy as np

def red(message,**kwargs): return termcolor.colored(str(message),color="red",attrs=[k for k,v in kwargs.items() if v is True])
def green(message,**kwargs): return termcolor.colored(str(message),color="green",attrs=[k for k,v in kwargs.items() if v is True])
def blue(message,**kwargs): return termcolor.colored(str(message),color="blue",attrs=[k for k,v in kwargs.items() if v is True])
def cyan(message,**kwargs): return termcolor.colored(str(message),color="cyan",attrs=[k for k,v in kwargs.items() if v is True])
def yellow(message,**kwargs): return termcolor.colored(str(message),color="yellow",attrs=[k for k,v in kwargs.items() if v is True])
def magenta(message,**kwargs): return termcolor.colored(str(message),color="magenta",attrs=[k for k,v in kwargs.items() if v is True])
def grey(message,**kwargs): return termcolor.colored(str(message),color="grey",attrs=[k for k,v in kwargs.items() if v is True])

class LOG:
    def __init__(self, ) -> None:
        pass

    def log_in(self, configs, project, name):
        self.base_path = f"exp/{project}/{name}"
        self.image_path = f"{self.base_path}/images"
        self.ckpts_path = f"{self.base_path}/ckpts"
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            os.makedirs(self.image_path)
            os.makedirs(self.ckpts_path)
        self.logger = wandb.init(dir=self.base_path, project=project, name=name, config=configs)

    def log_out(self):
        self.logger.finish(quiet=True)

    def info(self, message):
        print(green(message, bold=True, underline=True))

    def warn(self, message):
        print(red(message, bold=True, underline=True))

    def log_metric_init(self):
        self.logger.define_metric("train/epoch")
        self.logger.define_metric("test/epoch")
        self.logger.define_metric("train/*", step_metric="train/epoch")
        self.logger.define_metric("test/*", step_metric="test/epoch")

    def log_value(self, tag, name, value, step):
        info = {}
        info[f"{tag}/{name}"] = value
        info[f"{tag}/epoch"] = step
        self.logger.log(info)

    def log_images(self, tag, name, values, step, n_col=None):
        images = []
        if n_col is not None:
            n_img,h_img,w_img = values.shape[0],values.shape[1],values.shape[2]
            n_row = n_img // n_col
            for i in range(n_col):
                sub_images = values[n_row*i:n_row*(i+1)].reshape([n_row*h_img,w_img,-1])
                images.append(sub_images)
            images = np.concatenate(images, axis=-2)
            images = wandb.Image(images)
        else:
            for i in range(len(values)):
                image = wandb.Image(values[i])
                images.append(image)

        info = {}
        info[f"{tag}/{name}"] = images
        info[f"{tag}/epoch"] = step
        self.logger.log(info)