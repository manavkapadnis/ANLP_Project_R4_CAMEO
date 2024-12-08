import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
from PIL import Image
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from ast import literal_eval

# Hugging Face imports
from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoImageProcessor, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from transformers.trainer_callback import EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

# Evaluation imports
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider

# Load and preprocess data
df = pd.read_csv("/home/mkapadni/work/anlp_project/data/MTL_train_new.csv")
df_val = pd.read_csv("/home/mkapadni/work/anlp_project/data/MTL_val.csv")

# Data preprocessing
def preprocess_dataframe(df):
    # Binary encoding for task features
    for col in ['TXT', 'OBJ', 'CNT', 'COL']:
        df[col] = df[col].apply(lambda x: 1 if x > 3 else 0)
    df.drop(['OTH'], axis=1, inplace=True)
    
    # Evaluate highest confidence items
    df['captions'] = df['captions'].apply(literal_eval)
    df['vqa_answers'] = df['vqa_answers'].apply(literal_eval)
    df['highest_confidence_caption'] = df['captions'].apply(lambda x: x[0])
    df['highest_confidence_answer'] = df['vqa_answers'].apply(lambda x: x[0])
    
    return df

df = preprocess_dataframe(df)
df_val = preprocess_dataframe(df_val)

# Prompt template
prompt = """You are a helpful vision assistant. Given an image, perform the following tasks:
1. **Generate a detailed caption** describing the image contents, including objects, actions, colors, text (if any), and other relevant details.

2. **Determine if the following skills are required** to answer each question below. For each question, output a binary value (`1` for Yes, `0` for No) for the following skills:
   - **Object Recognition is needed to answer the question (ObjRec)**  
   - **Text Recognition is needed to answer the question (TextRec)**  
   - **Color Recognition is needed to answer the question (ColorRec)**  
   - **Counting is needed to answer the question (Count)**  

3. **Answer the question** directly in 1-3 words based on the image and caption.

### Format your output:
Caption:{Generated Caption}<SEP>ObjRec: {1 or 0}<SEP>TextRec: {1 or 0}<SEP>ColorRec: {1 or 0}<SEP>Count: {1 or 0}<SEP>Answer: {Direct Answer in 1-3 Words}

### Example Input:
Question: What is the color of the apples?
Image of a group of red apples in a basket with a price tag reading "$5".
  
### Example Output:
Caption: A basket of red apples with a price tag reading $5.<SEP>ObjRec:1<SEP>TextRec:1<SEP>ColorRec:1<SEP>Count:1<SEP>Answer:Red<END>
"""

class MultiTaskVQAModel:
    def __init__(self, args):
        # Vision Encoder
        self.vision_processor = AutoImageProcessor.from_pretrained(args.vision_model)
        self.visual_encoder = AutoModel.from_pretrained(args.vision_model, trust_remote_code=True)
        
        # Freeze or apply LoRA to vision encoder
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"]
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
        elif args.freeze_vm:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        
        # Language Model
        self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Apply LoRA to LLM
        if args.llm_use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                r=args.llm_r, 
                lora_alpha=args.llm_alpha, 
                lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
        
        # Projection layer
        self.llama_proj = nn.Linear(self.visual_encoder.config.hidden_size, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        
        self.args = args
        self.prompt = prompt

    def prepare_features(self, examples):
        # Prepare image features
        images = [Image.open(os.path.join(self.args.image_dir, img)) for img in examples['IMG']]
        image_features = self.vision_processor(images=images, return_tensors="pt")['pixel_values']
        
        # Prepare text features
        captions = examples['highest_confidence_caption']
        final_strings = [
            f"Caption: {caption}<SEP>ObjRec:{obj}<SEP>TextRec:{txt}<SEP>ColorRec:{col}<SEP>Count:{cnt}<SEP>Answer:{answer}<END>"
            for caption, obj, txt, col, cnt, answer in zip(
                captions, 
                examples['OBJ'], 
                examples['TXT'], 
                examples['COL'], 
                examples['CNT'], 
                examples['highest_confidence_answer']
            )
        ]
        
        return {
            'image_features': image_features,
            'input_text': final_strings
        }

    def compute_metrics(self, eval_pred):
        # Implement evaluation metrics similar to the original score method
        predictions, labels = eval_pred
        
        # Convert predictions and labels to text
        pred_texts = self.llama_tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_texts = self.llama_tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Create dictionaries for scoring
        ref = {str(i): [ref_text] for i, ref_text in enumerate(label_texts)}
        hypo = {str(i): [pred_text] for i, pred_text in enumerate(pred_texts)}
        
        # Compute scores
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        
        final_scores = {}
        for scorer, method in scorers:
            score, _ = scorer.compute_score(ref, hypo)
            if isinstance(score, list):
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        
        return final_scores

def main(args):
    # Initialize the model
    model = MultiTaskVQAModel(args)
    
    # Prepare datasets
    train_dataset = Dataset.from_pandas(df)
    val_dataset = Dataset.from_pandas(df_val)
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=args.savedmodel_path,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(args.savedmodel_path, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
    )
    
    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=model.llama_tokenizer, 
        mlm=False
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model.llama_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=model.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    trainer.train()
    
    # Save the model
    trainer.save_model(os.path.join(args.savedmodel_path, 'final_model'))

if __name__ == '__main__':
    # Assuming you have a parser set up similarly to the original script
    args = parser.parse_args()
    seed_everything(3407, workers=True)
    main(args)