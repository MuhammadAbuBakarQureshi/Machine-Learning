from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def tensorboard_to_dataframe(value, scalar_events):

    return pd.DataFrame([
        {'epoch': event.step, value: event.value}
        for event in scalar_events
    ])

def get_fit_data(log_dir):

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # print(f"Available scalars: {ea.Tags()['scalars']}")

    train_loss = ea.Scalars('Loss/train')
    test_loss = ea.Scalars('Loss/test')
    train_accuracy = ea.Scalars('Accuracy/train')
    test_accuracy = ea.Scalars('Accuracy/test')


    train_loss = tensorboard_to_dataframe(value = 'train_loss', scalar_events=train_loss)
    train_accuracy = tensorboard_to_dataframe(value = 'train_accuracy', scalar_events=train_accuracy)
    test_loss = tensorboard_to_dataframe(value = 'test_loss', scalar_events=test_loss)
    test_accuracy = tensorboard_to_dataframe(value = 'test_accuracy', scalar_events=test_accuracy)

    model_results = train_loss.merge(test_loss, how='right')
    model_results = model_results.merge(train_accuracy, how='right')
    model_results = model_results.merge(test_accuracy, how='right')

    return model_results
