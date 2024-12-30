from model import TextClassifier
import torch.nn
import torch
from transformers import AutoTokenizer

num_classes = 3
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


device = 'cuda'
model_save_path = 'text_classifier.pt'
model = TextClassifier(model_name, num_classes)
model.load_state_dict(torch.load(model_save_path))
model = model.to(device)
model.eval()

def predict(texts, tokenizer, model):
    model.eval()
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=1)
    return predictions.cpu().numpy()

test_texts = [
    "why is Finance minister of india is imposing heavy taxes on citizens",
    "troll him even more",
    'you are a racist',
    'he sleeps in class',
    'all islam invaders should be deported'
]

predictions = predict(test_texts, tokenizer, model)
print("Predictions:", predictions)