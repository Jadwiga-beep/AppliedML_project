import torch
import torch.nn as nn
from CNN import CNN
import copy


EPOCHS     = 50
BATCH_SIZE = 32
PATIENCE   = 5
LR         = 1e-3


def train(X_train, y_train, X_val, y_val, num_classes, input_shape, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.tensor(X_train).float().to(device)
    X_val = torch.tensor(X_val).float().to(device)
    y_train = torch.tensor(y_train).long().to(device)
    y_val = torch.tensor(y_val).long().to(device)

    model = CNN(input_shape=input_shape, num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        X_train, y_train = X_train[perm], y_train[perm]

        total_loss = 0.0
        for i in range(0, len(X_train), BATCH_SIZE):
            xb, yb = X_train[i:i+BATCH_SIZE], y_train[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
            val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()

        print(f"[{name}] Epoch {epoch+1}/{EPOCHS} — loss: {total_loss/len(X_train):.4f}  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[{name}] Early stopping.")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, X_test, y_test, name):
    device = next(model.parameters()).device

    X_test = torch.tensor(X_test).float().to(device)
    y_test = torch.tensor(y_test).long().to(device)

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    print(f"[{name}] Test accuracy: {test_acc:.4f}")
    return test_acc