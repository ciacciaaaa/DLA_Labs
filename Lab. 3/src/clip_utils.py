import torch
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader  
from tqdm import tqdm
#

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

def get_processor(model_id="openai/clip-vit-base-patch16"):
    return CLIPProcessor.from_pretrained(model_id)

def collate_images(batch, processor):
    """
    Prepara un batch di immagini e label adatto per l'addestramento/valutazione con CLIP.
    """
    images = [x["image"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    inputs = processor(images=images, return_tensors="pt")
    return inputs, labels


def zeroshot_clip(dataset, model_id="openai/clip-vit-base-patch16", batch_size=64):
    """
    Valuta CLIP in zero-shot sul dataset di validazione.
    Il dataset deve avere 'train' (per le classi) e 'validation'.
    """
    
    classnames = dataset["train"].features["label"].names
    model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
    processor = get_processor(model_id)

    # Prepara text embeddings (un prompt per classe)
    with torch.no_grad():
        text_inputs = processor(
            text=[f"a photo of a {c}" for c in classnames],
            return_tensors="pt", padding=True
        ).to(DEVICE)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    # Validation loader
    loader = DataLoader(dataset["validation"], batch_size=batch_size,
                        shuffle=False, collate_fn=lambda b: collate_images(b, processor))

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            img_embs = model.get_image_features(**inputs)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

            logits = img_embs @ text_embs.T
            preds = logits.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def finetune_clip(dataset, model_id="openai/clip-vit-base-patch16",
                  batch_size=32, lr=1e-4, epochs=3, train_text_encoder=False):
    """
    Fine-tune CLIP in maniera parameter-efficient:
    - train_text_encoder=False -> fine-tuning solo image encoder
    - train_text_encoder=True -> fine-tuning anche text encoder
    """
    classnames = dataset["train"].features["label"].names
    model = CLIPModel.from_pretrained(model_id).to(DEVICE)
    processor = get_processor(model_id)

    # Freeze tutti i parametri
    for param in model.parameters():
        param.requires_grad = False
    # Sblocca image encoder
    for param in model.vision_model.parameters():
        param.requires_grad = True
    # Sblocca text encoder solo se richiesto
    if train_text_encoder:
        for param in model.text_model.parameters():
            param.requires_grad = True

    loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True,
                        collate_fn=lambda batch: collate_images(batch, processor))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Prepara text embeddings
    with torch.no_grad():
        text_inputs = processor(
            text=[f"a photo of a {c}" for c in classnames],
            return_tensors="pt", padding=True
        ).to(DEVICE)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            img_embs = model.get_image_features(**inputs)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

            logits = img_embs @ text_embs.T
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    return model

def evaluate_clip(model, dataset, batch_size=64):
    """
    Valuta il modello fine-tunato sul dataset di validazione.
    """
    classnames = dataset["train"].features["label"].names
    processor = get_processor()
    with torch.no_grad():
        text_inputs = processor(
            text=[f"a photo of a {c}" for c in classnames],
            return_tensors="pt", padding=True
        ).to(DEVICE)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    loader = DataLoader(dataset["validation"], batch_size=batch_size,
                        shuffle=False, collate_fn=lambda b: collate_images(b, processor))

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            img_embs = model.get_image_features(**inputs)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
            logits = img_embs @ text_embs.T
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

