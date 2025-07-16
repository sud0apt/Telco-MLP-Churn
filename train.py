import torch
from torch import nn
from torch.utils.data import DataLoader
import csv

def train_model(model, train_dataset, test_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Initialize log file
    with open("training_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Train_Accuracy", "Test_Accuracy"])

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for x_cat, x_num, y in train_loader:
            output = model(x_cat, x_num)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, pred = torch.max(output, 1)
            correct_train += (pred == y).sum().item()
            total_train += y.size(0)

        train_acc = 100.0 * correct_train / total_train
        test_acc = evaluate_model(model, test_loader, return_acc=True)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # Log metrics
        with open("training_metrics.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, total_loss, train_acc, test_acc])


def evaluate_model(model, dataloader, return_acc=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_cat, x_num, y in dataloader:
            output = model(x_cat, x_num)
            _, pred = torch.max(output, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total
    if return_acc:
        return acc
    else:
        print(f"Test Accuracy: {acc:.2f}%")
