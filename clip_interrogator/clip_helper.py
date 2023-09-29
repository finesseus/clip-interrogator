import torch
from PIL.Image import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPHelper:
    def __init__(self, pretrained_model: str = "openai/clip-vit-large-patch14") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(pretrained_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model)
        self.pretrained_model = pretrained_model

    def change_model(self, pretrained_model: str):
        self.model = CLIPModel.from_pretrained(pretrained_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model)
        self.pretrained_model = pretrained_model

    def get_image_features(self, image: Image) -> torch.FloatTensor:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features

    def get_text_features(self, text: str) -> torch.FloatTensor:
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features

    def calculate_similarity(self, image_embeds: torch.FloatTensor, text_embeds: torch.FloatTensor) -> Tuple[
        torch.FloatTensor, torch.FloatTensor]:
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        return logits_per_text, logits_per_image

    def text_img_scores(self, image_embeds: torch.FloatTensor, text_embeds: torch.FloatTensor) -> torch.Tensor:
        logits_per_text, logits_per_image = self.calculate_similarity(image_embeds, text_embeds)
        probs = logits_per_text.softmax(dim=1)
        return probs
