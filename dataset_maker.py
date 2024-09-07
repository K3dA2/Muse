import torch
import numpy as np
from tqdm import tqdm
import os
from model import VQVAE 
from utils import get_data_loader, count_parameters, save_img_tensors_as_grid
import matplotlib.pyplot as plt

def data_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, max_grad_norm=1.0, epoch_start=0, save_img=True, show_img=False):
    indices_list = []
    with torch.no_grad():
        for epoch in range(epoch_start, n_epochs):
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit='batch')
            for imgs, _ in progress_bar:
                imgs = imgs.to(device)
                indices = model.return_indices(imgs)
                out,_ = model(imgs)
                indices_list.append(indices.cpu().numpy())

                '''
                print(indices[0].unsqueeze(0).shape)
                with torch.no_grad():
                    img = model.decode(indices[0].unsqueeze(0))
                out = out[0].unsqueeze(0).detach()
                #img = model.decoder(c)
                img = img[0].unsqueeze(0).detach()

                if torch.equal(out,img):
                    print("yes")
                else:
                    print("difference: ", (out - img).mean())
                #img = out
                img = img.squeeze().permute(1, 2, 0).cpu().numpy()
                mean=[0.7002, 0.6099, 0.6036]
                std=[0.2195, 0.2234, 0.2097]
                mean = np.array(mean)
                std = np.array(std)
                img = img * std + mean
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)

                plt.imshow(img)
                plt.axis('off')
                plt.show()
                '''
        return np.concatenate(indices_list)

if __name__ == "__main__":
    path = '/Users/ayanfe/Documents/Datasets/Waifus/Train'
    val_path = '/Users/ayanfe/Documents/Datasets/MNIST Upscaled/Val'
    model_path = 'weights/waifu-vqvae_epoch.pth'
    
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = VQVAE(latent_dim = 64, num_embeddings=512, beta=0.25, use_ema=False, e_width=64,d_width=64)  # Assuming Unet is correctly imported and defined
    model.to(device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.MSELoss().to(device)
    print(f"Model parameters: {count_parameters(model)}")

    data_loader = get_data_loader(path, batch_size=64)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    indices = data_loop(
        n_epochs=1,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=data_loader,
        epoch_start=0,
    )

    np.save('indices.npy', indices)
    print("Indices saved to indices.npy")

    # Save indices to a text file
    np.savetxt('indices.txt', indices, fmt='%d')
    print("Indices saved to indices.txt")
