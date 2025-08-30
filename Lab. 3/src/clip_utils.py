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


class CLIPForClassification(nn.Module):
    def __init__(self, clip_model, num_labels):
        super().__init__()
        self.clip = clip_model
        self.num_labels = num_labels
        # testuale: class embeddings
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_labels, bias=False)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        # Otteniamo feature immagine
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output  # [batch, hidden_dim]

        # Proiettiamo nello spazio CLIP
        image_embeds = self.clip.visual_projection(pooled)

        # Classificazione
        logits = self.classifier(image_embeds)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}