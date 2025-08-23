import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW
import requests
from tqdm import tqdm

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Mapping WordNet ID -> human label
# ---------------------------
words_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(words_url)
wnid_to_name_full = response.json()
wnid_to_name = {wnid: name for (_, (wnid, name)) in wnid_to_name_full.items()}

def create_readable_id2label(dataset):
    original_id2label = {id: label for id, label in enumerate(dataset["train"].features['label'].names)}
    readable_id2label = {id: wnid_to_name.get(wnid, wnid) for id, wnid in original_id2label.items()}
    return readable_id2label

def add_human_labels(dataset, readable_id2label):
    def get_human_label(example):
        return {"human_label": readable_id2label[example["label"]]}
    return dataset.map(get_human_label)

# ---------------------------
# Processor getter
# ---------------------------
def get_processor(model_id="openai/clip-vit-base-patch16"):
    return CLIPProcessor.from_pretrained(model_id)

# ---------------------------
# Collate function per DataLoader
# ---------------------------
def collate_human_labels(batch, processor, class2id):
    images = [ex["image"] for ex in batch]
    labels = torch.tensor([class2id[ex["human_label"]] for ex in batch], dtype=torch.long)
    inputs = processor(images=images, return_tensors="pt")
    return inputs, labels

# ---------------------------
# Funzione zero-shot batch
# ---------------------------
def zeroshot_clip(dataset, model_id="openai/clip-vit-base-patch16", batch_size=64):
    if "human_label" not in dataset["train"].column_names:
        raise ValueError("La colonna 'human_label' non è presente nel dataset. Esegui il mapping prima.")

    classnames = dataset["train"].unique("human_label")
    class2id = {c: i for i, c in enumerate(classnames)}

    model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
    processor = get_processor(model_id)

    with torch.no_grad():
        text_inputs = processor(
            text=[f"a photo of a {c}" for c in classnames],
            return_tensors="pt", padding=True
        ).to(DEVICE)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_human_labels(batch, processor, class2id)
    )

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

# ---------------------------
# Funzione Fine-tuning parameter-efficient (LoRA-style)
# ---------------------------
def finetune_clip(dataset, model_id="openai/clip-vit-base-patch16",
                  batch_size=32, lr=1e-4, epochs=3, train_text_encoder=False):
    """
    Fine-tune CLIP in maniera parameter-efficient:
    - train_text_encoder=False -> fine-tuning solo image encoder
    - train_text_encoder=True -> fine-tuning anche text encoder
    """
    if "human_label" not in dataset["train"].column_names:
        raise ValueError("La colonna 'human_label' non è presente nel dataset. Esegui il mapping prima.")

    classnames = dataset["train"].unique("human_label")
    class2id = {c: i for i, c in enumerate(classnames)}

    model = CLIPModel.from_pretrained(model_id).to(DEVICE)
    processor = get_processor(model_id)

    # Freeze tutti i parametri tranne quelli selezionati (parameter-efficient)
    for param in model.parameters():
        param.requires_grad = False

    # Sblocca image encoder
    for param in model.vision_model.parameters():
        param.requires_grad = True
    # Sblocca text encoder solo se richiesto
    if train_text_encoder:
        for param in model.text_model.parameters():
            param.requires_grad = True

    # Loader train
    loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_human_labels(batch, processor, class2id)
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Embedding testuali
    with torch.no_grad():
        text_inputs = processor(
            text=[f"a photo of a {c}" for c in classnames],
            return_tensors="pt", padding=True
        ).to(DEVICE)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    loss_fn = torch.nn.CrossEntropyLoss()

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
