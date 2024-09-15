import torch
from model import VQVAE 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
import datetime
import os
import torch.nn.utils as utils
from transformer import Muse, Config
from torchvision import transforms
import numpy as np
from utils import count_parameters
import json
from clip import CLIP,CLIPConfig

class ImageTagDataset(Dataset):
    def __init__(self, json_file, image_folder, file_extension='.jpg', transform=None):
        # Load the JSON file with image names and token lists
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.image_folder = image_folder
        self.file_extension = file_extension
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the image name and corresponding tokens from the JSON data
        image_name = list(self.data.keys())[idx]
        tokens = self.data[image_name]  # Tokens are already encoded and stored in the JSON file
        
        # Construct the image file path
        image_path = os.path.join(self.image_folder, f"{image_name}{self.file_extension}")
        
        # Open the image and convert it to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply the transformation to the image (if any)
        if self.transform:
            image = self.transform(image)

        # Convert tokens to a PyTorch tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        return image, tokens_tensor  # Return image and the token tensor


def training_loop(n_epochs, optimizer, model, clip, vqvae,
                  device, data_loader, valid_loader, 
                  max_grad_norm=1.0, epoch_start=0, mask_token=515):
    model.train()
    best_loss_valid = float('inf')
    for epoch in range(epoch_start, n_epochs):
        loss_train = 0.0
        loss_valid = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit='batch')
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)
            c = clip.txt_encoder(y)
            x = vqvae.return_indices(x)

            mask = torch.bernoulli(0.5 * torch.ones(x.shape, device=x.device))
            mask = mask.round().to(dtype=torch.int64).to(x.device)

            mask_tokens = torch.ones(x.shape[0], 1, device=x.device).long() * mask_token
            mask_indices = mask * x + (1 - mask) * mask_tokens
            mask_indices = mask_indices.long()  # Ensure mask_indices is of type torch.long

            if np.random.random() <= 0.1:
                _, loss = model(mask_indices,y=x)
            else:
                _, loss = model(mask_indices,c,x)

            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            loss_train += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        model.eval()
        with torch.no_grad():
            for valid_tensors, tkns in valid_loader:
                valid_tensors = valid_tensors.to(device)
                valid_tensors = vqvae.return_indices(valid_tensors)
                tkns = tkns.to(device)
                c = clip.txt_encoder(tkns)

                mask = torch.bernoulli(0.5 * torch.ones(valid_tensors.shape, device=valid_tensors.device))
                mask = mask.round().to(dtype=torch.int64).to(valid_tensors.device)

                mask_tokens = torch.ones(valid_tensors.shape[0], 1, device=valid_tensors.device).long() * mask_token
                mask_indices = mask * valid_tensors + (1 - mask) * mask_tokens
                mask_indices = mask_indices.long()  # Ensure mask_indices is of type torch.long
                
                if np.random.random() <= 0.1:
                     _, valid_loss = model(mask_indices,y = valid_tensors)
                else:
                    _, valid_loss = model(mask_indices, c,valid_tensors)

                
                loss_valid += valid_loss.item()
        
        loss_valid /= len(valid_loader)

        if loss_valid < best_loss_valid:
            best_loss_valid = loss_valid
            model_filename = 'vqvae-transformer.pth'
            model_path = os.path.join('weights', model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

        with open("vqvae-sampler.txt", "a") as file:
            file.write(f"{loss_train / len(data_loader)}\n")

        with open("vqvae-val-sampler.txt", "a") as file:
            file.write(f"{loss_valid}\n")

        print('{} Epoch {}, Training loss {}, Validation loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(data_loader), loss_valid))

        model.train()
    

if __name__ == "__main__":
    json_file = '/Users/ayanfe/Documents/Datasets/Waifus/train.json'
    image_folder = '/Users/ayanfe/Documents/Datasets/Waifus/Train'
    valid_file = '/Users/ayanfe/Documents/Datasets/Waifus/val.json'
    clip_model_path = 'weights/cnn-clip-mask_false.pth'
    vq_model_path = 'weights/waifu-vqvae_epoch.pth'
    model_path = 'weights/vqvae-transformer.pth'
    epoch = 0

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    
    config = Config()
    model = Muse(config)
    model.to(device)

    clip_config = CLIPConfig(vocab_len=50257, seq_len=82, 
                        latent_dim=256,use_mask=False)
    clip = CLIP(clip_config)
    clip.to(device)

    vqvae = VQVAE(latent_dim = 64, num_embeddings=512, beta=0.25, use_ema=False, e_width=64,d_width=64)  # Assuming Unet is correctly imported and defined
    vqvae.to(device)
    vqvae.eval()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    #Freeze and load weights
    clip_checkpoint = torch.load(clip_model_path)
    clip.load_state_dict(clip_checkpoint['model_state_dict'])

    vqvae_checkpoint = torch.load(vq_model_path)
    vqvae.load_state_dict(vqvae_checkpoint['model_state_dict'])

    for param in clip.parameters():
        param.requires_grad = False
    
    for param in vqvae.parameters():
        param.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.6589, 0.6147, 0.6220), (0.2234, 0.2234, 0.2158))
    ])

    print(f"Number of parameters: {count_parameters(model)}")
    # Initialize the dataset and dataloader
    dataset = ImageTagDataset(json_file=json_file, image_folder=image_folder, file_extension='.jpg', transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    val_dataset = ImageTagDataset(json_file=valid_file, image_folder=image_folder, file_extension='.jpg', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    
    # Optionally load model weights if needed
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    
    training_loop(
        n_epochs=300,
        optimizer=optimizer,
        model=model,
        clip=clip,
        vqvae=vqvae,
        device=device,
        data_loader=data_loader,
        valid_loader=val_loader,
        epoch_start=epoch + 1,
        mask_token=512
    )
