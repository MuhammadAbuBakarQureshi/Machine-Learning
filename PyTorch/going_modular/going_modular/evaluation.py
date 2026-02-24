import torch
import random
import matplotlib.pyplot as plt
from PIL import Image

def make_predictions(model,
                     image,
                     classes,
                     device):

    model.to(device)
    image = image.to(device)

    logits = model(image.unsqueeze(dim=0))
    pred_prob = torch.softmax(logits, dim=1)
    pred = pred_prob.argmax(dim=1).item()
    pred_prob = pred_prob.squeeze(dim=0)

    confidence = pred_prob[pred]
    pred_label = classes[pred]

    return pred_label, confidence


def plot_predictions(model,
                     image_path_list,
                     transform,
                     k: int,
                     classes,
                     device):

    image_path_list = random.sample(image_path_list, k=k)

    model.eval()

    with torch.inference_mode():

        rows = int(k/3) + 1
        cols = 3

        plt.figure(figsize=(20, int(k*1.5)))

        for i, image in enumerate(image_path_list):

            true_label = image.parent.stem
            image = Image.open(image)

            transform_image = transform(image)

            pred_label, confidence = make_predictions(model,
                                                      transform_image,
                                                      classes,
                                                      device)

            title = f"Pred: {pred_label} | True: {true_label} | {confidence}"

            plt.subplot(rows, cols, i + 1)

            if true_label == pred_label:

                plt.title(label=title, c='g')
            else: 
                plt.title(label=title, c='r')

            plt.imshow(image)
            plt.axis(False)
