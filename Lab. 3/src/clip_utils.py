import torch
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader  
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

    #label_column = "human_label" if "human_label" in dataset["train"].column_names else "label"
    #classnames = dataset["train"].unique(label_column)
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


def zero_shot_classification_with_plots(dataset_split, model, processor, device, id2label, n_plot=15):
    class_labels = list(id2label.values())
    text_prompts = [f"an image of a {label}" for label in class_labels]

    # Processing of texts
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    correct_predictions = 0
    total_images = len(dataset_split)

    # per plotting
    examples_for_plot = []

    for i, example in enumerate(tqdm(dataset_split, desc="Evaluating Zero-Shot Performance")):
        image = example['image']
        true_label_id = example['label']

        # Processing of images
        image_inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)

        # Normalization
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T)
        probs = similarity.softmax(dim=-1)

        prediction = torch.argmax(probs).item()

        if prediction == true_label_id:
            correct_predictions += 1

        # Examples for plotting
        if len(examples_for_plot) < n_plot:
            examples_for_plot.append({
                "image": image,
                "true_label": id2label[true_label_id],
                "pred_label": id2label[prediction],
                "correct": prediction == true_label_id
            })

    accuracy = correct_predictions / total_images

    # Plot
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for ax, ex in zip(axes.flat, examples_for_plot):
        ax.imshow(ex["image"], cmap="gray")
        color = "green" if ex["correct"] else "red"
        ax.set_title(f"True: {ex['true_label']}\nPred: {ex['pred_label']}", 
                     fontsize=22, color=color)
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()

    return accuracy
