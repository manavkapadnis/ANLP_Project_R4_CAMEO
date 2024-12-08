import pandas as pd
import numpy as np
from pprint import pprint
from ast import literal_eval
from lightning_tools.callbacks import add_callbacks
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoImageProcessor, AutoModel
import torch.nn as nn
import pandas as pd
from PIL import Image
from lightning.pytorch import seed_everything
import os
import json
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
import torch
from configs.config import parser


df_train = pd.read_csv("/home/mkapadni/work/anlp_project/data/MTL_train_new.csv")
df_val = pd.read_csv("/home/mkapadni/work/anlp_project/data/MTL_val.csv")
# df.head()

# make the TXT, OBJ, CNT, COL columns as if value more than 3, then 1 else 0 and drop OTH column
df_train['TXT'] = df_train['TXT'].apply(lambda x: 1 if x > 3 else 0)
df_train['OBJ'] = df_train['OBJ'].apply(lambda x: 1 if x > 3 else 0)
df_train['CNT'] = df_train['CNT'].apply(lambda x: 1 if x > 3 else 0)
df_train['COL'] = df_train['COL'].apply(lambda x: 1 if x > 3 else 0)
df_train.drop(['OTH'], axis=1, inplace=True)
df_val['TXT'] = df_val['TXT'].apply(lambda x: 1 if x > 3 else 0)
df_val['OBJ'] = df_val['OBJ'].apply(lambda x: 1 if x > 3 else 0)
df_val['CNT'] = df_val['CNT'].apply(lambda x: 1 if x > 3 else 0)
df_val['COL'] = df_val['COL'].apply(lambda x: 1 if x > 3 else 0)
df_val.drop(['OTH'], axis=1, inplace=True)

df_train['captions'] = df_train['captions'].apply(literal_eval)
df_train['vqa_answers'] = df_train['vqa_answers'].apply(literal_eval)
df_val['captions'] = df_val['captions'].apply(literal_eval)
df_val['vqa_answers'] = df_val['vqa_answers'].apply(literal_eval)

df_train['highest_confidence_caption'] = df_train['captions'].apply(lambda x: x[0])
df_train['highest_confidence_answer'] = df_train['vqa_answers'].apply(lambda x: x[0])
df_val['highest_confidence_caption'] = df_val['captions'].apply(lambda x: x[0])
df_val['highest_confidence_answer'] = df_val['vqa_answers'].apply(lambda x: x[0])

# Creating Dataloader for train and validation dataset


# vision_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
# text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Function to process images
# def process_image(image_path):
#     image = Image.open(image_path)
#     inputs = vision_processor(images=image, return_tensors="pt")
#     return inputs['pixel_values']

# # Function to process text
# def process_text(text):
#     inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     return inputs['input_ids'], inputs['attention_mask']

# sample 100 of each
# df_train = df_train.sample(100).reset_index(drop=True)
# df_val = df_val.sample(100).reset_index(drop=True)


class VQADataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.vision_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
        # self.vision_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

        # self.text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Process image
        image_path = os.path.join(self.image_dir, row['IMG'])
        image = Image.open(image_path)
        image_features = self.vision_processor(images=image, return_tensors="pt").pixel_values[0]
        
        # Process text (question)
        # question_input_ids, question_attention_mask = self._process_text(row['QSN'])
        question = row['QSN']
        
        # Extract other features from the dataframe
        txt_feature = row['TXT']
        obj_feature = row['OBJ']
        col_feature = row['COL']
        cnt_feature = row['CNT']
        highest_confidence_caption = row['highest_confidence_caption']
        highest_confidence_answer = row['highest_confidence_answer']
        # give final string like this -> Caption: A basket of red apples with a price tag reading $5.<SEP>ObjRec:1<SEP>TextRec:1<SEP>ColorRec:1<SEP>Count:1<SEP>Answer:Red<END>
        final_string = f"Answer:{highest_confidence_answer}<END>"
        
        return {
            'image_id': row['IMG'],
            'image': image_features,
            'question': question,
            # 'question_attention_mask': question_attention_mask,
            # 'txt': txt_feature,
            # 'obj': obj_feature,
            # 'col': col_feature,
            # 'cnt': cnt_feature,
            # 'caption': highest_confidence_caption,
            'answer': final_string
        }

def create_datasets(args):
    train_dataset = VQADataset(df_train, args.image_dir)
    val_dataset = VQADataset(df_val, args.image_dir)
    # test_dataset = VQADataset(pd.read_csv(args.test_csv), args.image_dir)
    return train_dataset, val_dataset


class DataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        # self.train_csv = args.train_csv
        # self.val_csv = args.val_csv
        # self.test_csv = test_csv

        self.image_dir = args.image_dir


    def setup(self, stage: str):
        # Load datasets for each stage if necessary
        if stage == "fit" or stage is None:
            # print(self.train_csv.shape)
            # print(self.val_csv.shape)
            self.train_dataset = VQADataset(df_train, self.image_dir + "/train")
            self.val_dataset = VQADataset(df_val, self.image_dir+"/val")
            

        # if stage == "test" or stage is None:
            # self.test_dataset = VQADataset(pd.read_csv(self.test_csv), self.image_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
prompt ="""You are a helpful vision assistant. Given a low quality input image, perform the following tasks:
1. **Answer the question** directly in 1-3 words based on the image and caption. 
Note: You can't respond as "Quality issues are too severe to recognize visual content." 

### Format your output:
Generate a single string output with the following structure:
Answer: {Direct Answer in 1-3 Words}

### Example Input:
Question: What is the color of the apples?
Image of a group of red apples in a basket with a price tag reading "$5".

### Example Output:
Answer:Red<END>
"""

class MultiTaskVQAModel(pl.LightningModule):
    """
    MultiTaskVQAModel model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = AutoModel.from_pretrained(args.vision_model,trust_remote_code=True)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        bos_token = "<SEP>"

        # Ensure the token exists in the vocabulary
        if bos_token not in self.llama_tokenizer.get_vocab():
            # If not, you might need to add it
            self.llama_tokenizer.add_special_tokens({'additional_special_tokens': [bos_token]})

        # Get the token ID for the chosen BOS token
        bos_token_id = self.llama_tokenizer.convert_tokens_to_ids(bos_token)

        # Assign this ID as the bos_token_id
        self.llama_tokenizer.bos_token_id = bos_token_id

        # Verify that it's set correctly
        print(f"BOS Token ID set to: {self.llama_tokenizer.bos_token_id}")
        
        if args.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype="auto",
                # load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype="auto",
            )
         
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLM LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLM Done')

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        # self.llama_proj = nn.Linear(768, self.llama_model.config.hidden_size)

        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = prompt
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            # (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores


    def encode_img(self, image):
        image_embeds = []
        # for image in images:
        device = image.device
        if self.hparams.global_only:
            image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
        else:
            image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
        image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama


    def prompt_wrap(self, img_embeds, atts_img, question):

        batch_size = img_embeds.shape[0]
        device = img_embeds.device
        max_length = 0

        # Lists to hold individual components for all samples
        p_before_embeds_list = []
        p_after_embeds_list = []

        # Process each sample in the batch
        for idx in range(batch_size):
            prompt = self.prompt  # Assuming 'text' contains the prompt for each sample
            prompt = f'{self.prompt} Input: <Img><ImageHere></Img> Question: {question[idx]}\nOutput: '
            p_before, p_after = prompt.split('<ImageHere>')  # Splitting based on your placeholder

            # Tokenize the parts of the prompt
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)

            # Embed the tokens and append to list
            p_before_embeds = self.embed_tokens(p_before_tokens.input_ids)
            p_after_embeds = self.embed_tokens(p_after_tokens.input_ids)
            p_before_embeds_list.append(p_before_embeds)
            p_after_embeds_list.append(p_after_embeds)

            # Keep track of the max sequence length for padding purposes
            seq_len = p_before_embeds.shape[1] + img_embeds.shape[1] + p_after_embeds.shape[1]
            if seq_len > max_length:
                max_length = seq_len

        # Initialize tensor to hold all embedded prompts
        wrapped_img_embeds = torch.zeros((batch_size, max_length, self.llama_model.config.hidden_size), device=device)
        wrapped_atts_img = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)

        # Fill in the tensor for each sample
        for idx, (before, after) in enumerate(zip(p_before_embeds_list, p_after_embeds_list)):
            seq_len = before.shape[1] + img_embeds[idx].shape[0] + after.shape[1]
            wrapped_img_embeds[idx, :seq_len] = torch.cat([before, img_embeds[idx].unsqueeze(0), after], dim=1)
            wrapped_atts_img[idx, :seq_len] = 1  # Set attention mask to 1 for non-padded values

        return wrapped_img_embeds, wrapped_atts_img


    def forward(self, samples):
        image = samples["image"]
        question = samples["question"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, question)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["answer"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        batch_size = img_embeds.shape[0]

        empty_targets = (
            # torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
            torch.ones([atts_img.shape[0], atts_img.shape[1]],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        pad = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.pad_token_id
        targets = torch.cat([empty_targets, targets, pad], dim=1)

        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            # labels=targets,
        )
        
        loss = nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
        
        if torch.isnan(loss):
            print(outputs.logits)
            
        # loss = outputs.loss
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['answer'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        # print("image", image.shape)
        img_embeds, atts_img = self.encode_img(image)
        # print("img_embeds", img_embeds.shape)
        # print("atts_img", atts_img.shape)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, samples["question"])

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["image_id"]})
        return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['answer'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, samples["question"])

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["image_id"]})
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()



def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    if args.ckpt_file is not None:
        model = MultiTaskVQAModel.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = MultiTaskVQAModel(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(3407, workers=True)
    train(args)


if __name__ == '__main__':
    main()