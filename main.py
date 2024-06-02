import torch
from datasets import get_dataloaders
from model import Net
from train import train_model, test_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=5)
    model = Net().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    train_model(model, train_loader, optimizer, loss_fn, device, num_epochs=10)
    test_loss, test_accuracy = test_model(model, test_loader, loss_fn, device)
    
    print('Test loss: {:.6f}, Test accuracy: {:.6f}'.format(test_loss, test_accuracy))

if __name__ == "__main__":
    main()
