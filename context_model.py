import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import SensorEncoder

class ContextEmbeddingModel(nn.Module):
    def __init__(self, embed_dim=64, llm_hidden_size=960, num_classes=6):
        super().__init__()
        self.encoder = SensorEncoder(embed_dim)
        self.projector = nn.Sequential(nn.Linear(64, 384), nn.ReLU(), nn.Linear(384, llm_hidden_size))
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        self.llm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        self.head = nn.Linear(llm_hidden_size, num_classes)
        for p in self.llm.parameters():
            p.requires_grad = False
        prefix_text = "Classify the activity as walking, walking upstairs, walking downstairs, sitting, standing, or laying.\n\nSensor context: "
        suffix_text = "\n\nActivity:"

        self.register_buffer("prefix_ids", self.tokenizer(prefix_text, return_tensors="pt")["input_ids"])
        self.register_buffer("suffix_ids", self.tokenizer(suffix_text, return_tensors="pt")["input_ids"])

    def get_embedding(self, x):
        sensor_emb = self.encoder(x)
        sensor_emb = self.projector(sensor_emb)
        return sensor_emb

    def classify_from_embedding(self, sensor_emb):
        batch_size = sensor_emb.shape[0]

        embedding_layer = self.llm.get_input_embeddings()
        prefix_embeds = embedding_layer(self.prefix_ids)
        suffix_embeds = embedding_layer(self.suffix_ids)

        prefix_embeds = prefix_embeds.expand(batch_size, -1, -1)
        suffix_embeds = suffix_embeds.expand(batch_size, -1, -1)

        sensor_emb = sensor_emb.unsqueeze(1)

        full_embeds = torch.cat([prefix_embeds, sensor_emb, suffix_embeds], dim=1)

        outputs = self.llm(inputs_embeds=full_embeds, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        last_token_hidden = last_hidden[:, -1, :]
        logits = self.head(last_token_hidden)

        return logits

    def forward(self, x):
        sensor_emb = self.get_embedding(x)
        logits = self.classify_from_embedding(sensor_emb)
        return logits

if __name__ == "__main__":
    model = ContextEmbeddingModel()
    # print("Loaded Successfully brrrrr")
    # print("Hidden size:", model.llm.config.hidden_size)
    # num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Trainable params:", num_trainable) 
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable)

    fake_ass_bitch = torch.randn(4, 128, 9) 
    logits = model(fake_ass_bitch)
    print("output shape:", logits.shape)
