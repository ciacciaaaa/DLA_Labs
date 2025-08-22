# src/trainer.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchmetrics import Accuracy
from sklearn.metrics import classification_report, confusion_matrix
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
import time

class Callback:
    """Classe base astratta per tutti i callback."""
    def on_train_begin(self, trainer): pass
    def on_epoch_end(self, trainer): pass
    def on_train_end(self, trainer): pass

class EarlyStopping(Callback):
    """
    Callback per interrompere l'addestramento se una metrica monitorata smette di migliorare.
    """
    def __init__(self, monitor='val_accuracy', patience=5, mode='max', min_delta=0.001):
        """
        Args:
            monitor (str): La metrica da monitorare (es. 'val_accuracy' o 'val_loss').
            patience (int): Numero di epoche da attendere senza miglioramenti prima di fermarsi.
            mode (str): 'max' per le metriche che devono aumentare (accuracy), 'min' per quelle che devono diminuire (loss).
            min_delta (float): Miglioramento minimo richiesto per considerare l'epoca come un progresso.
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.wait_counter = 0
        self.best_score = -np.inf if mode == 'max' else np.inf
        print(f"EarlyStopping enabled. Monitoring '{monitor}' with patience={patience}.")

    def on_epoch_end(self, trainer):
        current_score = trainer.history[self.monitor][-1]
        
        improved = False
        if self.mode == 'max':
            if current_score > self.best_score + self.min_delta:
                improved = True
        else: # mode == 'min'
            if current_score < self.best_score - self.min_delta:
                improved = True
        
        if improved:
            self.best_score = current_score
            self.wait_counter = 0 # Resetta il contatore perché c'è stato un miglioramento
        else:
            self.wait_counter += 1 # Nessun miglioramento, incrementa il contatore
        
        # Se il contatore raggiunge la pazienza, ferma il training
        if self.wait_counter >= self.patience:
            print(f"\nEarlyStopping triggered after {self.patience} epochs with no improvement.")
            trainer._stop_training = True # Imposta il flag per fermare il loop di fit



    
import torch
from torchmetrics.classification import Accuracy
from tqdm import tqdm
import wandb


class Trainer:
    """
    Trainer class with training, validation, and testing logic.
    Supports CPU, CUDA, MPS, Early Stopping, and optional wandb logging.
    """

    def __init__(self, model, train_dl, val_dl, optimizer, criterion, num_classes,
                 device='cpu', callbacks=None,
                 enable_wandb=False, wandb_config=None):

        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_classes = num_classes
        self.device = device
        self.history = {'loss': [], 'val_accuracy': [], 'val_loss': []}
        
        self.callbacks = callbacks if callbacks is not None else []
        self._stop_training = False

        # wandb
        self.enable_wandb = enable_wandb
        self.wandb_config = wandb_config if wandb_config is not None else {}
        if self.enable_wandb:
            wandb.init(**self.wandb_config)
            wandb.watch(self.model, log="all", log_freq=100)

    def _train_epoch(self, epoch_num):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_dl, desc=f'Training Epoch {epoch_num}', leave=False)
        for xs, ys in progress_bar:
            xs, ys = xs.to(self.device), ys.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(xs)
            loss = self.criterion(logits, ys)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_dl)
        self.history['loss'].append(avg_loss)

        if self.enable_wandb:
            wandb.log({"train/loss": avg_loss, "epoch": epoch_num})

        return avg_loss

    def _evaluate(self, epoch_num):
        self.model.eval()
        accuracy_metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        total_val_loss = 0.0

        with torch.no_grad():
            for xs, ys in tqdm(self.val_dl, desc='Evaluating', leave=False):
                xs, ys = xs.to(self.device), ys.to(self.device)
                logits = self.model(xs)
                loss = self.criterion(logits, ys)
                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                accuracy_metric.update(preds, ys)
        
        avg_val_loss = total_val_loss / len(self.val_dl)
        accuracy = accuracy_metric.compute().item()
        
        self.history['val_loss'].append(avg_val_loss)
        self.history['val_accuracy'].append(accuracy)

        if self.enable_wandb:
            wandb.log({
                "val/loss": avg_val_loss,
                "val/accuracy": accuracy,
                "epoch": epoch_num
            })
        
        return avg_val_loss, accuracy

    def fit(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs on device '{self.device}'...")
        self._stop_training = False
        
        for cb in self.callbacks: 
            cb.on_train_begin(self)
            
        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_acc = self._evaluate(epoch)
            
            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            for cb in self.callbacks: 
                cb.on_epoch_end(self)
            
            if self._stop_training:
                print(f"Training stopped early at epoch {epoch}.")
                break
        
        for cb in self.callbacks: 
            cb.on_train_end(self)

        if self.enable_wandb:
            wandb.finish()
            
        print("\n--- Training Complete ---")
        return self.history

    def test(self, test_dl):
        """
        Valuta il modello sul test set.

        Args:
            test_dl (DataLoader): DataLoader del test set.
            log_wandb (bool, optional): Se True logga i risultati su wandb.

        Returns:
            dict: {'test_loss': ..., 'test_accuracy': ...}
        """
        self.model.eval()
        accuracy_metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        total_test_loss = 0.0

        with torch.no_grad():
            for xs, ys in tqdm(test_dl, desc='Testing', leave=False):
                xs, ys = xs.to(self.device), ys.to(self.device)
                logits = self.model(xs)
                loss = self.criterion(logits, ys)

                total_test_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                accuracy_metric.update(preds, ys)

        avg_test_loss = total_test_loss / len(test_dl)
        test_accuracy = accuracy_metric.compute().item()

        results = {
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
        }

        print(f"\n--- Test Results ---\nLoss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.4f}")

        return results
    
    def save_model(self, output_dir: str, suffix: str = "final") -> str:
        """
        Save model and return saving filepath

        """
        os.makedirs(output_dir, exist_ok=True)
        model_name = self.model.__class__.__name__
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{model_name}_{suffix}_{timestamp}.pth"
        
        filepath = os.path.join(output_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        
        print(f"Model saved to {filepath}")
        return filepath

    


def plot_history(history):
    """
    Visualizes the learning and validation curves saved in the 'history' dictionary.

    """
    # Check if history is empty
    if not history or not history['loss']:
        print("History is empty. Cannot plot.")
        return
        
    epochs = range(1, len(history['loss']) + 1)

    #plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Training and Validation Loss
    ax1.plot(epochs, history['loss'], 'o--', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'o-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_xticks(epochs)

    # Validation Accuracy
    ax2.plot(epochs, history['val_accuracy'], 'o-', label='Validation Accuracy', color='orange')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_xticks(epochs)

    plt.tight_layout()
    plt.show()


# =========================================================================================

from sklearn.metrics import classification_report
from torchmetrics import Accuracy

def test_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, 
               criterion: torch.nn.Module, device: str, num_classes: int):
    """
    Performs a final evaluation of a trained model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): The DataLoader for the test set.
        criterion (torch.nn.Module): The loss function (e.g., nn.CrossEntropyLoss).
        device (str): The device to run the evaluation on ('cpu', 'mps', 'cuda').
        num_classes (int): The number of classes for metric calculation.

    Returns:
        dict: A dictionary containing the final test loss, accuracy, and classification report.
    """
    print("\n--- Starting Final Evaluation on Test Set ---")
    model.to(device)
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

    total_test_loss = 0.0
    all_preds = []
    all_gts = []
    
    # Initialize torchmetrics Accuracy
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for xs, ys in tqdm(test_loader, desc="Testing"):
            xs, ys = xs.to(device), ys.to(device)

            # Forward pass
            logits = model(xs)
            loss = criterion(logits, ys)
            total_test_loss += loss.item()

            # Get predictions
            preds = torch.argmax(logits, dim=1)

            # Update metrics
            accuracy_metric.update(preds, ys)
            all_preds.append(preds.cpu())
            all_gts.append(ys.cpu())

    # Calculate final metrics
    avg_loss = total_test_loss / len(test_loader)
    final_accuracy = accuracy_metric.compute().item()
    
    # Concatenate all batches
    gts_np = torch.cat(all_gts).numpy()
    preds_np = torch.cat(all_preds).numpy()

    # Generate classification report
    #report_str = classification_report(gts_np, preds_np, digits=3)
    #report_dict = classification_report(gts_np, preds_np, digits=3, output_dict=True)

    # Print results
    print(f"\nAverage Test Loss: {avg_loss:.4f}")
    print(f"Final Test Accuracy: {final_accuracy * 100:.2f}%")
    print("\nClassification Report:", classification_report(gts_np, preds_np, digits=3))
    

