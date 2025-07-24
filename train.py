from models.dit import MFDiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlow
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import time
import os


if __name__ == '__main__':
    n_steps = 50000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 48
    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')

    # dataset = torchvision.datasets.CIFAR10(
    #     root="cifar",
    #     train=True,
    #     download=True,
    #     transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
    # )
    dataset = torchvision.datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=T.Compose([T.Resize((32, 32)), T.ToTensor(),]),
    )

    

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    

    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=1,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    meanflow = MeanFlow(channels=1,
                        image_size=32,
                        num_classes=10,
                        flow_ratio=0.50,
                        time_dist=['lognorm', -0.4, 1.0],
                        cfg_ratio=0.10,
                        cfg_scale=2.0,
                        # experimental
                        cfg_uncond='u')

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i
    train_dataloader = cycle(train_dataloader)
    
    global_step = 0.0
    losses = 0.0
    mse_losses = 0.0

    log_step = 100
    sample_step = 100

    #summary_writer = SummaryWriter("runs/fm_mnist")
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()
        for step in pbar:
            data = next(train_dataloader)
            x = data[0].to(accelerator.device)
            c = data[1].to(accelerator.device)

            loss, mse_val = meanflow.loss(model, x, c)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()
            mse_losses += mse_val.item()

            if accelerator.is_main_process:
                if global_step % log_step == 0:
                    current_time = time.asctime(time.localtime(time.time()))
                    batch_info = f'Global Step: {global_step}'
                    loss_info = f'Loss: {losses / log_step:.6f}    MSE_Loss: {mse_losses / log_step:.6f}'

                    # Extract the learning rate from the optimizer
                    lr = optimizer.param_groups[0]['lr']
                    lr_info = f'Learning Rate: {lr:.6f}'

                    log_message = f'{current_time}\n{batch_info}    {loss_info}    {lr_info}\n'

                    with open('log.txt', mode='a') as n:
                        n.write(log_message)
                    #summary_writer.add_scalar('Loss/train', losses / log_step, global_step)
                    #summary_writer.add_scalar('MSE_Loss/train', mse_losses / log_step, global_step)
                    losses = 0.0
                    mse_losses = 0.0

            if global_step % sample_step == 0:
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, 'module') else model
                    z = meanflow.sample_each_class(model_module, 10)
                    log_img = make_grid(z, nrow=10)
                    img_save_path = f"images/step_{global_step}.png"
                    save_image(log_img, img_save_path)
                    #summary_writer.add_image('image', log_img, global_step)
                accelerator.wait_for_everyone()
                model.train()
                
    if accelerator.is_main_process:
        ckpt_path = f"checkpoints/step_{global_step}.pt"
        accelerator.save(model_module.state_dict(), ckpt_path)