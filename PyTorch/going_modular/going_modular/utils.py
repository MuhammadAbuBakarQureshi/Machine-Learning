
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_name = model_name if (model_name.endswith(".pt") or model_name.endswith(".pth")) else model_name + ".pt"

    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)

def print_train_time(start, end, device):

    time = end - start

    day = 0
    hour = 0
    min = 0
    sec = time

    while sec >= 60:

        if hour >= 24:

            day += 1
            hour -= 24

        elif sec >= 3600:

            hour += 1
            sec -= 3600

        elif time >= 60:

            min += 1
            sec -= 60

    day_output = f"{day} day{'s' if day > 1 else ''} : "
    hour_output = f"{hour} hour{'s' if hour > 1 else ''} : "
    min_output = f"{min} min{'s' if min > 1 else ''} : "
    sec_output = f"{sec} sec{'s' if sec > 1 else ''}"

    time_output = f"{day_output if day != 0 else ''}{hour_output if hour != 0 else ''}{min_output if min != 0 else ''}{sec_output if sec != 0 else ''}"

    print(f"Total time taken on {device} is {time_output}")
