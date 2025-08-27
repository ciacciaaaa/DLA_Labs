# Deep Learning Applications - Laboratory 3

## Overview:

This laboratory is designed as a hands-on introduction to Transformers and their applications to natural language processing (NLP) and vision-language tasks. The goal is to gain both a practical understanding of how to use pretrained models and a deeper intuition of how fine-tuning works in real scenarios.

Throughout the exercises, we progressively build up from simple baselines to more advanced training and evaluation pipelines:

1. Stable Baseline with __DistilBERT__:
    * Use ```DistilBERT``` as a feature extractor to obtain ```CLS``` embeddings.
    * Train a simple classifier (e.g., SVM) on top of the frozen representations.
    * Evaluate results to establish a solid baseline performance.

2. __Fine-tuning__ DistilBERT for Sequence Classification:
    * Tokenize the dataset splits with HuggingFace ```Datasets``` utilities.
    * Load a pretrained DistilBERT with a classification head (```AutoModelForSequenceClassification```).
    * Fine-tune the model using HuggingFace’s ```Trainer``` API.
    * Compare performance improvements against the baseline.

3. Exploring Vision-Language Models with __CLIP__:
    * Evaluate a pretrained CLIP model (zero-shot) on multiple datasets such as ImageNette, TinyImageNet, and another dataset of choice.
    * Investigate different adaptation strategies (image encoder only, text encoder, or both).
    * Apply parameter-efficient fine-tuning to squeeze out additional performance gains.

## Project Structure


## Exercises
### Exercise 1: Sentiment Analisys BERT
In this exercise we explore how to build a sentiment analysis model using a pre-trained BERT transformer. The goal is to classify movie reviews as either positive or negative and the chosen dataset is the _Cornell Rotten Tomatoes movie review_ dataset, which contains 5,331 positive and 5,331 negative processed sentences from Rotten Tomatoes movie reviews.

### Exercise 1.1:
The first step focused on data exploration through the HuggingFace library. The dataset is already conveniently split into three balanced subsets — training, validation, and test — each containing short sentences extracted from movie reviews. Every sentence is annotated with a binary label that identifies its sentiment as either positive or negative. To familiarise ourselves with the data, we loaded the three splits and inspected random samples, checking both the raw text and its corresponding label. This exploratory step ensures a clear understanding of the dataset before moving on to tokenisation and model fine-tuning in the subsequent phases of the exercise.

| Sentence       |Label |
| ------------- |---------- |
| _the vivid lead performances sustain interest and empathy , but the journey is far more interesting than the final destination ._   |1 (Positive)        |
| _feels like the grittiest movie that was ever made for the lifetime cable television network_ |0 (Negative)      |

### Exercise 1.2:

The second step of the exercise introduced the model that will serve as the backbone of our sentiment analysis pipeline: DistilBERT. This transformer is a compact version of BERT, trained with a teacher–student approach where the original BERT acts as the teacher.

We loaded both the model and its associated tokenizer using the HuggingFace ```AutoModel``` and ```AutoTokenizer``` classes. The tokenizer transforms raw text into the numerical tokens required by the model, handling tasks such as subword splitting and padding. To test the setup, we applied the tokenizer to a few sample sentences from the Rotten Tomatoes dataset and passed the resulting token IDs through the DistilBERT model. The outputs consisted of contextual embeddings for each token, as well as a pooled representation for the entire sentence. This exploratory step allowed us to confirm that the model and tokenizer were correctly configured, providing the foundation for the fine-tuning process that will follow.

```python
idx = 870
input = tokenizer(ds_train[idx]['text'], padding=True, return_tensors="pt")
output = model(**input)

print("Sample phrase:", ds_train['text'][idx])
print("Token IDs:", input["input_ids"], "\n")
print(output)
```
```output
{Sample phrase: the story , like life , refuses to be simple , and the result is a compelling slice of awkward emotions .
Token IDs: tensor([[  101,  1996,  2466,  1010,  2066,  2166,  1010, 10220,  2000,  2022,
          3722,  1010,  1998,  1996,  2765,  2003,  1037, 17075, 14704,  1997,
          9596,  6699,  1012,   102]]) 

BaseModelOutput(last_hidden_state=tensor([[[-0.1782, -0.1238, -0.0865,  ..., -0.0867,  0.4100,  0.3362],
         [-0.2704, -0.0790, -0.4201,  ...,  0.1000,  0.8274, -0.1355],
         [ 0.0728, -0.2583, -0.1176,  ..., -0.1747,  0.1437, -0.2213],
         ...,
         [-0.4658,  0.1920,  0.0862,  ..., -0.2875,  0.1009, -0.1235],
         [ 0.7563,  0.0812, -0.2714,  ...,  0.3315, -0.5503, -0.4448],
         [-0.5857, -0.1106,  0.3320,  ...,  0.0919,  0.3582, -0.0365]]],
       grad_fn=<NativeLayerNormBackward0>), hidden_states=None, attentions=None)}

```
### Exercise 1.3: a stable baseline
In this step, the goal was to establish a stable baseline by using DistilBERT as a feature extractor and then training a simple classifier on the extracted features.

To do so, we relied on the ```[CLS]``` token embeddings. In transformer models, such as BERT and DistilBERT, the ```[CLS]``` token is a special symbol added at the beginning of every input sequence. Its hidden representation in the final layer is designed to capture a global summary of the entire sentence, making it particularly useful for classification tasks. Instead of using all token embeddings, we extracted only the vector corresponding to the ```[CLS]``` token from the last hidden state of DistilBERT.

We implemented a helper function ```get_cls_embeddings```, which processes batches of text, encodes them with the tokenizer, and passes them through the model. For each sentence, the ```[CLS]``` embedding was extracted and used as the feature vector. These embeddings were then stacked into matrices for the training, validation, and test splits.

Once we obtained the embeddings, we trained a Linear SVM classifier (from Scikit-learn) on the training set. The validation and test splits were then used to measure performance. 

The evaluation reported both accuracy and a full classification report (precision, recall, F1) on the test set, giving us a first solid baseline. This result will serve as a benchmark for the next exercises, where we will move beyond feature extraction and experiment with fine-tuning DistilBERT end-to-end.

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.79      | 0.81   | 0.80     | 533     |
| 1     | 0.81      | 0.78   | 0.80     | 533     |
| **Accuracy** |       |        | **0.80** | **1066** |
| **Macro avg** | 0.80 | 0.80   | 0.80     | 1066    |
| **Weighted avg** | 0.80 | 0.80   | 0.80     | 1066    |

- **Validation accuracy:** 0.822  
- **Test accuracy:** 0.798

### Exercise 2.1: Token Preprocessing
The first step in preparing our dataset for a transformer model is __tokenization__. Our raw dataset returns a dictionary with strings, but the model expects input token IDs along with attention masks to properly process sequences.

We leverage the HuggingFace ```Dataset.map``` method to efficiently and lazily tokenize our datasets. This method applies a function to each element of the dataset, returning new fields such as:

* ```input_ids```: the numeric IDs representing each token in the text
* ```attention_mask```: a binary mask indicating which tokens are real vs padding

``` python
idx = 0
print('Before processing:', ds_train[idx].keys())
print('After processing:', tokenized_ds_train[idx].keys())
```

``` output
Before processing: dict_keys(['text', 'label'])
After processing: dict_keys(['text', 'label', 'input_ids', 'attention_mask'])
```

### Exercise 2.2: Setting up the Model to be Fine-tuned

Then the base model for is prepared for fine-tuning on a sentiment classification task. While DistilBERT is a general-purpose transformer trained with self-supervised objectives, we now adapt it for sequence classification. This involves attaching a new, randomly initialized classification head on top of the ```[CLS]``` token representation. This classification head is a small feed-forward network that maps the pooled representation to class probabilities.

Fortunately, HuggingFace provides a convenient implementation through the class ```AutoModelForSequenceClassification```. By instantiating the model this way, we automatically obtain DistilBERT with an additional classification layer, ready to be fine-tuned.

### Exercise 2.3: Fine-tuning DistilBERT

In this exercise, we fine-tuned DistilBERT on the Rotten Tomatoes dataset using HuggingFace’s ```Trainer```.  To prepare the training setup, we first instantiate a ```DataCollatorWithPadding```. This ensures that all sequences in a batch are padded to the same length dynamically, which is essential for efficient GPU training:

We then implemented a custom evaluation function that computed balanced accuracy, F1 score, precision, and recall. These metrics provided a comprehensive overview of the model’s classification ability beyond simple accuracy, especially in cases where class imbalance could affect performance. The results are stored in ```results```.

The training arguments defined the main hyperparameters of the experiment, including learning rate, batch size, number of epochs, and weight decay for regularization. In addition, we enabled checkpoint saving and evaluation at the end of each epoch, and we set up early stopping to prevent overfitting.

Finally, the Trainer was instantiated with the model, the tokenized training and validation datasets, the tokenizer, the data collator, the evaluation function, and the early stopping callback. The model was then fine-tuned on the training set and evaluated on the validation split.

The final evaluation metrics are summarized below:
| Metric            | Validation |
| ----------------- | ---------- |
| Balanced Accuracy |0.85        |
| F1 Score          |0.85        |
| Precision         |0.85        |
| Recall            |0.83        |

### Exercise 3.2: Fine-tuning a CLIP Model

In this exercise we explored the use of a pre-trained CLIP model ```openai/clip-vit-base-patch16``` for image classification. CLIP, trained on a very large collection of image–text pairs, has the remarkable ability to perform zero-shot classification: it can recognize classes without additional training by comparing images to textual prompts such as _“a photo of a {class}”_.

To better understand CLIP’s generalization abilities, we evaluated the model in the zero-shot setting on multiple datasets of different difficulty:

* __ImageNette__: a simplified 10-class subset of ImageNet. From HuggingFace's ```Sijuade/ImageNette```

* __Tiny ImageNet__: a 200-class dataset with smaller, lower-resolution images, making the classification task much harder. From HuggingFace's ```slegroux/tiny-imagenet-200-clean```

* __Dataset 3 (to be defined)__: chosen to explore performance in a different image domain or distribution, in order to test CLIP’s robustness beyond ImageNet-like datasets.

For each dataset, we prepared a set of text prompts corresponding to the class labels (e.g., __“a photo of a {class}”)_, extracted embeddings for both the images and the prompts, and classified each image by cosine similarity to the text embeddings. This allowed us to compare CLIP’s zero-shot performance across domains.


| Dataset       |n. classes |Zero-shot Accuracy (%)|
| ------------- |---------- |-------------|
| ImageNette    |10         |98.46     |
| Tiny Imagenet |200        |57.14     |
| Dataset 3.    |    ?      |     ?    |

During zero-shot experiments CLIP achieved very high accuracy on ImageNette (98.46%) but significantly lower performance on Tiny ImageNet (57.14%). For Tiny ImageNet the original WordNet IDs labels (e.g., n02106662, n01770393) were converted to human-interpretable labels, so the low performance is not due to unintelligible class names. CLIP performs extremely well on ImageNette (and ImageNet-like datasets) because it was pretrained on large-scale web image-text pairs, which cover general object categories with clear and descriptive labels, closely matching the classes in ImageNette. In contrast, Tiny ImageNet contains 64×64 pixel images, much smaller than those CLIP saw during pretraining, and includes 200 fine-grained, visually similar classes, making zero-shot classification much more challenging even with descriptive labels. These factors explain why zero-shot accuracy is substantially lower on Tiny ImageNet compared to ImageNette

After establishing these baselines, we fine-tuned CLIP (with parameter-efficient methods) to evaluate how much additional performance could be gained compared to the zero-shot setup. Fine-tuning experiments will focus on unfreezing only the image encoder (and optionally the text encoder) to keep computational costs low.
