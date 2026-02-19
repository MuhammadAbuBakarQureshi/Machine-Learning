
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from timeit import default_timer as timer
from tqdm.autonotebook import tqdm

from .utils import print_train_time

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim,
               train_accuracy_fn,
               BATCH_SIZE: int,
               device: torch.device):

    total_train_loss, total_train_accuray = 0, 0
    model.to(device)

    train_accuracy_fn.reset()

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        model.train()

        logits = model(X)
        train_pred = torch.argmax(logits, dim=1)
        train_loss = loss_fn(logits, y)

        total_train_loss += train_loss
        train_accuracy_fn.update(train_pred, y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if batch % 2 == 0:

            print(f"samples remaning: {batch*BATCH_SIZE} / {len(dataloader)*BATCH_SIZE}")

    total_train_loss /= len(dataloader)
    total_train_accuray = train_accuracy_fn.compute()

    print(f"Train loss: {total_train_loss} --- | Train Accuracy: {total_train_accuray}")

    return total_train_loss, total_train_accuray


def test_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               test_accuracy_fn,
               device: torch.device):

    total_test_loss, total_test_accuray = 0, 0
    model.to(device)

    model.eval()

    with torch.inference_mode():

        test_accuracy_fn.reset()

        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            logits = model(X)
            test_pred = torch.argmax(logits, dim=1)
            test_loss = loss_fn(logits, y)

            total_test_loss += test_loss
            test_accuracy_fn.update(test_pred, y)

        total_test_loss /= len(dataloader)
        total_test_accuray = test_accuracy_fn.compute()

        print(f"Test loss: {total_test_loss} --- | Test Accuracy: {total_test_accuray}")

    return total_test_loss, total_test_accuray


def fit_fn(model: nn.Module,
           train_dataloader: torch.utils.data.DataLoader,
           test_dataloader: torch.utils.data.DataLoader,
           loss_fn: nn.Module,
           optimizer: torch.optim,
           train_accuracy_fn,
           test_accuracy_fn,
           batch_size,
           epochs: int,
           summary_write_path: str,
           device: torch.device):

    start_time = timer()

    writer = SummaryWriter(summary_write_path)

    for epoch in tqdm(range(epochs)):

        train_loss, train_accuracy = train_step(model=model,
                   dataloader=train_dataloader,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   train_accuracy_fn=train_accuracy_fn,
                   BATCH_SIZE=batch_size,
                   device=device)

        test_loss, test_accuracy = test_step(model=model,
                   dataloader=test_dataloader,
                   loss_fn=loss_fn,
                   test_accuracy_fn=test_accuracy_fn,
                   device=device)


        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)


    writer.close()

    end_time = timer()

    print_train_time(start=start_time,
                     end=end_time,
                     device=device)
