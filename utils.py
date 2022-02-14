import random, time

import wandb


def train(project="st", entity=None, epochs=10, bar=None):
    run = wandb.init(
        # Set the project where this run will be logged
        project=project, 
        entity=entity,
        # Track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": epochs,
        })

    # This simple block simulates a training loop logging metrics
    offset = random.random() / 5
    for epoch in range(1, epochs+1):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        # 2️⃣ Log metrics from your script to W&B
        wandb.log({"acc": acc, "loss": loss})
        time.sleep(0.1)
        bar.progress(epoch/epochs)
        
    # Mark the run as finished
    wandb.finish()