import torch
import torch.nn.functional as F
import math
from transformer import Muse, Config
from model import VQVAE
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from clip import CLIP,CLIPConfig

def gamma_func(mode="cosine"):
    if mode == "linear":
        return lambda r: 1 - r
    elif mode == "cosine":
        return lambda r: torch.cos(torch.tensor(r * math.pi / 2))
    elif mode == "square":
        return lambda r: 1 - r ** 2
    elif mode == "cubic":
        return lambda r: 1 - r ** 3
    else:
        raise NotImplementedError

def mask_by_random_topk(mask_len, probs, temperature=1.0, device="cpu"):
    gumbel_noise = torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(device)
    confidence = torch.log(probs) + temperature * gumbel_noise
    sorted_confidence, _ = torch.sort(confidence, dim=-1)
    
    mask_len_expanded = mask_len.unsqueeze(-1)
    
    cut_off = torch.take_along_dim(sorted_confidence, mask_len_expanded.to(torch.long), dim=-1)
    
    masking = (confidence < cut_off)
    return masking

def sample(text, T, model, clip ,cfg_scale = 5,mode='cosine', seq_len=256, mask_token=512, device='cpu'):
    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Set a padding token (use eos_token or define a new one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    # Tokenize text using the GPT-2 tokenizer
    text_tokens = tokenizer(text, padding='max_length', max_length=82, return_tensors='pt')['input_ids']
    
    # Replace padding tokens with space tokens
    space_token_id = tokenizer.encode(" ", add_special_tokens=False)[0]
    padding_token_id = tokenizer.pad_token_id
    
    # Replace padding tokens with space tokens in the tokenized sequences
    text_tokens[text_tokens == padding_token_id] = space_token_id

    text_tokens = text_tokens.to(device)

    c = clip.txt_encoder(text_tokens)
    
    gamma = gamma_func(mode)
    
    pred = torch.ones((1, seq_len), dtype=torch.long, device=device) * mask_token
    
    unknown_number_in_the_beginning = torch.sum(pred == mask_token, dim=-1).to(torch.float32)
   

    for t in range(T):
        logits, _ = model(pred,c = c)
        if cfg_scale > 0:
            u_logits, _ = model(pred)
            logits = torch.lerp(u_logits.to("cpu"), logits.to("cpu"), cfg_scale).to(device)
        
        sampled_indices = torch.distributions.categorical.Categorical(logits=logits).sample()
        
        unknown_map = (pred == mask_token)
        
        sampled_indices = torch.where(unknown_map, sampled_indices, pred)
       
        ratio = 1. * (t + 1) / T
        mask_ratio = gamma(ratio)

        probs = F.softmax(logits, dim=-1)
        
        selected_probs = torch.gather(probs, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)

        selected_probs = torch.where(unknown_map, selected_probs, torch.tensor([float('inf')], device=device))

        min_value = torch.tensor(0.0, device=unknown_number_in_the_beginning.device, dtype=unknown_number_in_the_beginning.dtype)

        mask_len = torch.floor(unknown_number_in_the_beginning * mask_ratio).clamp(min=min_value, max=unknown_number_in_the_beginning-1)

        masking = mask_by_random_topk(mask_len, selected_probs, temperature=(4.5 * (1. - ratio)), device=device)

        pred = torch.where(masking, mask_token, sampled_indices)

    return pred

if __name__ == '__main__':
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    config = Config()
    model = Muse(config)
    model.to(device)

    vqvae = VQVAE(latent_dim = 64, num_embeddings=512, beta=0.25, use_ema=False, e_width=64,d_width=64) 
    vqvae.to(device)

    clip_config = CLIPConfig(vocab_len=50257, seq_len=82, 
                        latent_dim=256,use_mask=False)
    clip = CLIP(clip_config)
    clip.to(device)

    model_path = 'weights/vqvae-transformer.pth'
    vq_model_path = 'weights/waifu-vqvae_epoch.pth'
    clip_model_path = 'weights/cnn-clip-mask_false.pth'

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    vq_checkpoint = torch.load(vq_model_path)
    vqvae.load_state_dict(vq_checkpoint['model_state_dict'])

    clip_checkpoint = torch.load(clip_model_path)
    clip.load_state_dict(clip_checkpoint['model_state_dict'])

    txt = ["1girl", "ahoge", "black_eyes", "confetti", "crying", "disembodied_hand", 
           "drill_hair", "feet_out_of_frame", "gloves", "hat", "hatsune_miku", "highres", 
           "holding", "holding_hat", "holding_unworn_clothes", "huan1034097", "kasane_teto",
            "long_sleeves", "mesmerizer_(vocaloid)", "open_mouth", "pants", "pink_hair", "pink_pants", "shirt", "smile", 
            "striped_clothes", "striped_shirt", "suspenders", "twin_drills", "two-tone_background", "utau", 
            "vocaloid", "white_background", "wide-eyed", "yellow_background", "yellow_gloves"]
    
    txt = ["1girl, ahoge, black_eyes, confetti, crying, disembodied_hand, drill_hair, feet_out_of_frame, gloves, hat, hatsune_miku, highres, holding, holding_hat, holding_unworn_clothes, huan1034097, kasane_teto, long_sleeves, mesmerizer_(vocaloid), open_mouth, pants, pink_hair, pink_pants, shirt, smile, striped_clothes, striped_shirt, suspenders, twin_drills, two-tone_background, utau, vocaloid, white_background, wide-eyed, yellow_background, yellow_gloves"]


    txt2 = ["1girl, white_hair, blue_eyes"]
    
    tkns = sample(txt2,8, model, clip, cfg_scale=7,device=device)
    #print(tkns)
    with torch.no_grad():
        img = vqvae.decode(tkns)

    img = img

    mean = [0.7002, 0.6099, 0.6036] 
    sd = [0.2195, 0.2234, 0.2097]
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(sd).view(1, 3, 1, 1).to(device)

    img = img * std + mean 
    img = img.squeeze().permute(1, 2, 0).cpu().numpy()

    plt.imshow(img)
    plt.axis('off')
    plt.show()
