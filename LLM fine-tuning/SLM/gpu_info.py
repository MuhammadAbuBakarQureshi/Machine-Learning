import torch

def get_gpu_info():

    if torch.cuda.is_available():
    
        device_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_idx)    
        total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
        gpu_name = gpu_name.replace(" ", "_")

    return (gpu_name, f"{total_vram:.2f}")


gpu_name, gpu_vram = get_gpu_info()

print(f"{gpu_name}_{gpu_vram}")