from clip_interrogator.clip_interrogator_hf import Config, Interrogator, LabelTable, load_list
from PIL import Image
from transformers import CLIPModel

# hf_clip_model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
hf_clip_model_name = "patrickjohncyh/fashion-clip"
open_clip_model_name = "ViT-H-14/laion2B-s32B-b79K"



image_path = 'jillian_model.png'
text = "pink mini dress with boning"
image = Image.open(image_path).convert('RGB')
ci = Interrogator(Config(clip_model_name=open_clip_model_name, clip_model_name_hf=hf_clip_model_name, device='cpu'))

# if_oc = ci.image_to_features(image)
# if_hf = ci.image_to_features_hf(image)
# tf_oc = ci.text_to_features([text])
# tf_hf = ci.text_to_features_hf([text])
# ci = Interrogator(Config(clip_model_name="hf-hub:patrickjohncyh/fashion-clip"))
table = LabelTable(load_list('dress_flavours.txt'), 'terms', ci)
best_match = table.rank(ci.image_to_features_hf(image), top_count=15)
prompt = ci.interrogate_hf(image, table, min_flavors=4, max_flavors=8, caption="")
print(prompt)
print(best_match)
