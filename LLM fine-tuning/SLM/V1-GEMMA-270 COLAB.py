import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
from transformers import pipeline

from trl import SFTConfig, SFTTrainer # type: ignore

from dotenv import load_dotenv
import os
import wandb

load_dotenv()


def sample_to_conversation_batch(batch):

    raw_prompts = []
    raw_completion = []
    messages = []

    for i in range(0, len(batch['sequence'])):

        prompt_dict = {"role": "user", "content": batch["sequence"][i]}
        completion_dict = {"role": "assistant", "content": batch["gpt-oss-120b-label-condensed"][i]}

        raw_prompts.append(prompt_dict)
        raw_completion.append(completion_dict)
        messages.append([prompt_dict, completion_dict])



    return {
        "raw_prompt": raw_prompts,
        "raw_completion": raw_completion,
        "messages": messages
    }

##################### ENVIRONMENT VARIABLES & CONFIGURATION #####################


MODEL_NAME = "google/gemma-3-270m-it"

DATASET_NAME = "mrdbourke/FoodExtract-1k"

CHECKPOINT_DIR = "./checkpoints_models"
BASE_LEARNING_RATE = 5e-5
BATCH_SIZE = 10
OPTIMIZER = "adamw_torch_fused"
METRIC_FOR_BEST_MODEL = "eval_loss"

WANDB_PROJECT = f"{MODEL_NAME}_{BASE_LEARNING_RATE}_{OPTIMIZER}_{METRIC_FOR_BEST_MODEL}_{BATCH_SIZE}_{DATASET_NAME}".replace("/", "_")

wandb.init(project=WANDB_PROJECT)
HF_TOKEN = os.getenv("HF_TOKEN")

##################### MODEL LOADING #####################

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager",
    token=HF_TOKEN
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)


print(f"[INFO] Model loaded: {MODEL_NAME}\n")

print(f"[INFO] Model using device: {model.device}")
print(f"[INFO] Model dtype: {model.dtype}")

###################### DATASET LOADING #####################




dataset = load_dataset(DATASET_NAME)

print(f"[INFO] Dataset loaded: {DATASET_NAME}\n")

print(f"[INFO] Number of samples in the dataset: {len(dataset['train'])}")


dataset = dataset.map(sample_to_conversation_batch,
                      batched=True)

dataset = dataset['train'].train_test_split(test_size=0.2,
                                       shuffle=False,
                                       seed=42)

print(f"[INFO] Number of samples in the training set: {len(dataset['train'])}")
print(f"[INFO] Number of samples in the test set: {len(dataset['test'])}")


######################## MODEL TRAINING CONFIGURATION #####################


torch_dtype = model.dtype


print(f"[INFO] Model training configuration initializing.\n")

print(f"[INFO] Using dtype: {torch_dtype}")
print(f"[INFO] Checkpoint directory: {CHECKPOINT_DIR}")
print(f"[INFO] Base learning rate: {BASE_LEARNING_RATE}")
print(f"[INFO] Batch size: {BATCH_SIZE}")


sft_config = SFTConfig(
    output_dir=CHECKPOINT_DIR,
    max_length=512,
    packing=False,
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    completion_only_loss=True,
    gradient_checkpointing=True,
    optim=OPTIMIZER,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=BASE_LEARNING_RATE,
    fp16=(torch_dtype == torch.float16),
    bf16=(torch_dtype == torch.bfloat16),
    load_best_model_at_end=True,
    metric_for_best_model=METRIC_FOR_BEST_MODEL,
    greater_is_better=False,
    lr_scheduler_type="constant",
    push_to_hub=False,
    report_to="wandb",
    run_name=WANDB_PROJECT
)


######################### MODEL TRAINING ########################


trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer
)

training_results = trainer.train()