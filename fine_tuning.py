import torch
from tqdm import tqdm
import numpy as np

from visualization import plot_training_curves

# feedforward without gradient updates
@torch.no_grad()
def feedforward(model, dataloader, device):
    model.eval()
    
    accuracies = []
    losses = []
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss, logits = outputs[0], outputs[1]
        
        # compute acc
        pred = torch.argmax(logits, dim=1)
        
        accuracies.append((pred == labels).sum().item() / logits.shape[0])
        losses.append(loss.item()) 
    # calculate the mean
    accuracies = np.mean(np.array(accuracies))
    losses = np.mean(np.array(losses))
    return losses, accuracies



# back propagation with gradient updates
def backpropagation(model, dataloader, device, optimizer):
    model.train()
    accuracies = []
    losses = []
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss, logits = outputs[0], outputs[1]
        
        # compute acc
        pred = torch.argmax(logits, dim=1)
        
        accuracies.append((pred == labels).sum().item() / logits.shape[0])
        losses.append(loss.item()) 
        
        loss.backward()
        optimizer.step()
    
    # calculate the mean
    accuracies = np.mean(np.array(accuracies))
    losses = np.mean(np.array(losses))
    
    return losses, accuracies
    
    
# model training loop
def model_finetuning(model, train_loader, valid_loader, device):
    
    learning_rate = 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    n_epochs = 10
    
    # get the initial statistics
    train_loss, train_acc = feedforward(model, train_loader, device)
    valid_loss, valid_acc = feedforward(model, valid_loader, device)
    print(f"Epoch 0/{n_epochs} | Train Acc: {train_acc:.3f} | Train Loss: {train_loss:.3f} | Valid Acc: {valid_acc:.3f} | Valid Loss: {valid_loss:.3f}")
    
    # training curves
    train_losses = [train_loss]
    train_accs = [train_acc]
    valid_losses = [valid_loss]
    valid_accs = [valid_acc]
    
    # Early Stopping criteria
    patience = 3
    not_improved = 0
    best_valid_loss = valid_loss
    threshold = 0.01
    
    # training epoches
    for epoch in range(n_epochs):
        # feedforward to estimate loss
        train_loss, train_acc = backpropagation(model, train_loader, device, optimizer)
        valid_loss, valid_acc = feedforward(model, valid_loader, device)
        
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Train Acc: {train_acc:.3f} | Train Loss: {train_loss:.3f} | Valid Acc: {valid_acc:.3f} | Valid Loss: {valid_loss:.3f}")
        
        # evaluate the current performance
        # strictly better
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            not_improved = 0
            # save the best model based on validation loss
            torch.save(model.state_dict(), f'{type(model).__name__}_finetuned.pth')
            # also save the optimizer state for future training
            torch.save(optimizer.state_dict(), f'{type(model).__name__}_finetuned_optimizer.pth')

        # becomes worse
        elif valid_loss > best_valid_loss + threshold:
            not_improved += 1
            if not_improved >= patience:
                print('Early Stopping Activated')
                break
            
    plot_training_curves(train_accs, train_loss, valid_accs, valid_loss)
    
    
        
        
        