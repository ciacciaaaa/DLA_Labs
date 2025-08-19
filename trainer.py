# src/trainer.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchmetrics import Accuracy
from sklearn.metrics import classification_report


class SimpleTrainer:
    """
    Class that contains the training and evaluation logic for a PyTorch model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_dl (DataLoader): DataLoader for the training set.
        val_dl (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer.
        num_classes (int): The number of classes in the classification problem.
        device (str, optional): The device to run computations on ('cpu', 'mps' or 'cuda'). Defaults to 'cpu'.
        use_amp (bool, optional): Whether to use Automatic Mixed Precision (AMP). Defaults to False.
    """
    def __init__(self, model, train_dl, val_dl, optimizer, num_classes, device='cpu', use_amp=False):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.device = device
        self.history = {'loss': [], 'val_accuracy': [], 'val_report': []}
        
        self.use_amp = use_amp and self.device == 'cuda'
        self.scaler = None
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP) on CUDA device.")
        elif use_amp and self.device == 'mps':
            print("Warning: AMP is requested but the device is 'mps'. "
                  "torch.cuda.amp is not available for MPS. Disabling AMP.")

    def _train_epoch(self, epoch_num):
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_dl, desc=f'Training Epoch {epoch_num}', leave=False)
        
        for xs, ys in progress_bar:
            xs, ys = xs.to(self.device), ys.to(self.device)
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                logits = self.model(xs)
                loss = F.cross_entropy(logits, ys)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(self.train_dl)
        self.history['loss'].append(avg_loss)
        return avg_loss

    def _evaluate(self):
        """Performs evaluation on the validation set."""
        self.model.eval()
        accuracy_metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        all_preds, all_gts = [], []

        with torch.no_grad():
            for xs, ys in tqdm(self.val_dl, desc='Evaluating', leave=False):
                xs, ys = xs.to(self.device), ys.to(self.device)
                
                with autocast(enabled=self.use_amp):
                    logits = self.model(xs)
                
                preds = torch.argmax(logits, dim=1)
                accuracy_metric.update(preds, ys)
                
                all_preds.append(preds.cpu())
                all_gts.append(ys.cpu())

        accuracy = accuracy_metric.compute().item()
        gts_np = torch.cat(all_gts).numpy()
        preds_np = torch.cat(all_preds).numpy()
        report = classification_report(gts_np, preds_np, zero_division=0, digits=3)
        
        self.history['val_accuracy'].append(accuracy)
        self.history['val_report'].append(report)
        return accuracy, report
        
    def fit(self, num_epochs):
        """
        Trains and evaluates the model for a specified number of epochs.

        Args:
            num_epochs (int): The total number of epochs for training.

        Returns:
            dict: The performance history (loss and accuracy).
        """
        print(f"Starting training for {num_epochs} epochs on device '{self.device}'...")
        for epoch in range(1, num_epochs + 1):
            avg_loss = self._train_epoch(epoch)
            val_acc, _ = self._evaluate()
            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")
            
        print("\n--- Final Validation Report ---")
        print(self.history['val_report'][-1])
        return self.history

def plot_history(history):
    """
    Function to plot the loss and accuracy saved in the trainer's history.
    """
    losses = history['loss']
    accs = history['val_accuracy']
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training Loss per Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(accs)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    best_acc = np.max(accs)
    best_epoch = np.argmax(accs)
    plt.title(f'Best Accuracy = {best_acc:.3f} @ epoch {best_epoch+1}')
    plt.show()

# This block is executed only if you run the file directly (e.g., `python src/trainer.py`)
# It's useful for running a small test to ensure the code works.
if __name__ == '__main__':
    # Test example
    print("Testing SimpleTrainer components...")
    
    # 1. Create dummy data
    dummy_train_ds = torch.utils.data.TensorDataset(torch.randn(100, 1, 28, 28), torch.randint(0, 10, (100,)))
    dummy_val_ds = torch.utils.data.TensorDataset(torch.randn(50, 1, 28, 28), torch.randint(0, 10, (50,)))
    train_loader = torch.utils.data.DataLoader(dummy_train_ds, batch_size=16)
    val_loader = torch.utils.data.DataLoader(dummy_val_ds, batch_size=16)

    # 2. Create a dummy model
    dummy_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28*28, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10)
    )

    # 3. Create an optimizer
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)

    # 4. Instantiate and run the trainer
    trainer = SimpleTrainer(dummy_model, train_loader, val_loader, optimizer, num_classes=10)
    history = trainer.fit(2) # Train for 2 epochs
    
    print("\nHistory:")
    print(history)
    # plot_history(history) # Uncomment to see the plot
    print("Test completed successfully.")