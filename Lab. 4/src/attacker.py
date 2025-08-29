# src/attacker.py

import torch
import torch.nn as nn
from tqdm.notebook import tqdm

class AdversarialAttackerFGSM:
    """
    Una classe per generare esempi avversari usando il metodo Iterative FGSM.
    """
    def __init__(self, model: nn.Module, criterion: nn.Module):
        """
        Args:
            model (nn.Module): Il modello da attaccare.
            criterion (nn.Module): La funzione di loss (es. CrossEntropyLoss).
        """
        self.model = model
        self.criterion = criterion
        self.device = next(model.parameters()).device

    def generate_batch_attack(self, x: torch.Tensor, y: torch.Tensor, epsilon: float = 8/255) -> torch.Tensor:
        """
        Genera un batch di esempi avversari con un singolo passo di FGSM (untargeted).
        """
        self.model.train()
        x_adv = x.clone().detach().to(self.device)
        y = y.to(self.device)
        
        x_adv.requires_grad = True
        
        output = self.model(x_adv)
        loss = self.criterion(output, y)
        self.model.zero_grad()
        loss.backward()
        
        grad_sign = x_adv.grad.data.sign()
        
        x_adv = x_adv + epsilon * grad_sign
        
        return torch.clamp(x_adv, -1, 1).detach()    

    def attack(self, x: torch.Tensor, y: torch.Tensor, 
               epsilon: float = 8/255, 
               max_steps: int = 20, 
               targeted: bool = False, 
               target_label: int = None):
        """
        Attacco avversario su un batch di immagini.

        """
        self.model.eval()
        x_adv = x.clone().detach().to(self.device)
        y = y.to(self.device)

        if targeted and target_label is None:
            raise ValueError("Target label must be provided for a targeted attack.")
        
        y_target = None
        if targeted:
            y_target = torch.tensor([target_label] * len(x), device=self.device)

        # Initial check
        initial_pred = self.model(x_adv).argmax(dim=1)
        if targeted and initial_pred.item() == y_target.item():
            print("Model already predicts the target label. No attack needed.")
            return x_adv, True, initial_pred.item()
        if not targeted and initial_pred.item() != y.item():
            print("Model is already incorrect on the original image. No attack needed.")
            return x_adv, True, initial_pred.item()

        for i in range(max_steps):
            x_adv.requires_grad = True
            output = self.model(x_adv)
            
            if targeted:
                loss = self.criterion(output, y_target)
            else:
                loss = self.criterion(output, y)

            self.model.zero_grad()
            loss.backward()
            grad_sign = x_adv.grad.data.sign()

            # Applica la perturbazione (update di FGSM)
            if targeted:
                # Muove immagine VERSO classe target (minimizzando la loss)
                x_adv = x_adv - epsilon * grad_sign
            else:
                # Muove immagine LONTANO dalla classe vera (massimizzando la loss)
                x_adv = x_adv + epsilon * grad_sign
            
            x_adv = torch.clamp(x_adv, -1, 1).detach()

            # Success check
            final_pred = self.model(x_adv).argmax(dim=1)
            if targeted and final_pred.item() == y_target.item():
                print(f"Targeted attack succeeded in {i+1} steps.")
                return x_adv, True, final_pred.item()
            if not targeted and final_pred.item() != y.item():
                print(f"Untargeted attack succeeded in {i+1} steps.")
                return x_adv, True, final_pred.item()

        print(f"Attack failed to fool the model within {max_steps} steps.")
        return x_adv, False, final_pred.item()