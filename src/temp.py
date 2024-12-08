import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoImageProcessor, AutoModel
import pandas as pd
import numpy as np
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader, TensorDataset

from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained(
    "microsoft/swin-base-patch4-window7-224",
)
model = AutoModel.from_pretrained(
    "microsoft/swin-base-patch4-window7-224",
    trust_remote_code=True,
)

inputs = processor(images=[image, image], return_tensors="pt")
print(inputs.keys())
print(inputs.pixel_values.shape)
outputs = model(**inputs)
print(outputs.keys())
print(outputs.last_hidden_state.shape)

print('num_features:', model.num_features)

# print("\n\nChecking Text LLM model\n\n")

#     model_name = "Qwen/Qwen2.5-3B-Instruct"

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype="auto",
#         device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# print(model_inputs.keys())

# df = pd.read_csv("/home/mkapadni/work/anlp_project/data/MTL_train_new.csv")
# df_val = pd.read_csv("/home/mkapadni/work/anlp_project/data/MTL_val.csv")
# # df.head()

# # make the TXT, OBJ, CNT, COL columns as if value more than 3, then 1 else 0 and drop OTH column
# df['TXT'] = df['TXT'].apply(lambda x: 1 if x > 3 else 0)
# df['OBJ'] = df['OBJ'].apply(lambda x: 1 if x > 3 else 0)
# df['CNT'] = df['CNT'].apply(lambda x: 1 if x > 3 else 0)
# df['COL'] = df['COL'].apply(lambda x: 1 if x > 3 else 0)
# df.drop(['OTH'], axis=1, inplace=True)
# df_val['TXT'] = df_val['TXT'].apply(lambda x: 1 if x > 3 else 0)
# df_val['OBJ'] = df_val['OBJ'].apply(lambda x: 1 if x > 3 else 0)
# df_val['CNT'] = df_val['CNT'].apply(lambda x: 1 if x > 3 else 0)
# df_val['COL'] = df_val['COL'].apply(lambda x: 1 if x > 3 else 0)
# df_val.drop(['OTH'], axis=1, inplace=True)

# df['captions'] = df['captions'].apply(literal_eval)
# df['vqa_answers'] = df['vqa_answers'].apply(literal_eval)
# df_val['captions'] = df_val['captions'].apply(literal_eval)
# df_val['vqa_answers'] = df_val['vqa_answers'].apply(literal_eval)

# df['highest_confidence_caption'] = df['captions'].apply(lambda x: x[0])
# df['highest_confidence_answer'] = df['vqa_answers'].apply(lambda x: x[0])
# df_val['highest_confidence_caption'] = df_val['captions'].apply(lambda x: x[0])
# df_val['highest_confidence_answer'] = df_val['vqa_answers'].apply(lambda x: x[0])


# vision_processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-224")
# text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# # Function to process images
# def process_image(image_path):
#     image = Image.open(image_path)
#     inputs = vision_processor(images=image, return_tensors="pt")
#     return inputs['pixel_values']

# # Function to process text
# def process_text(text):
#     inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     return inputs['input_ids'], inputs['attention_mask']


# class VQADataset(Dataset):
#     def __init__(self, dataframe):
#         self.dataframe = dataframe

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         row = self.dataframe.iloc[idx]
#         image_features = process_image(row['IMG'])
#         question_input_ids, question_attention_mask = process_text(row['QSN'])
        
#         # Extract other features from the dataframe as needed
#         txt_feature = row['TXT']
#         obj_feature = row['OBJ']
#         col_feature = row['COL']
#         cnt_feature = row['CNT']
#         highest_confidence_caption = row['highest_confidence_caption']
#         highest_confidence_answer = row['highest_confidence_answer']
        
#         return {
#             'image': image_features.squeeze(0),
#             'question_input_ids': question_input_ids.squeeze(0),
#             'question_attention_mask': question_attention_mask.squeeze(0),
#             'txt': txt_feature,
#             'obj': obj_feature,
#             'col': col_feature,
#             'cnt': cnt_feature,
#             'caption': highest_confidence_caption,
#             'answer': highest_confidence_answer
#         }
    

# def create_dataloader(dataframe, batch_size=16):
#     dataset = VQADataset(dataframe)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return dataloader

# train_loader = create_dataloader(df)
# val_loader = create_dataloader(df_val)

# print("Data Loaders created successfully!")



