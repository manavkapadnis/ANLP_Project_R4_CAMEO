import pandas as pd
import numpy as np
import pickle
import json
from openai import OpenAI
from tqdm import tqdm
from huggingface_hub import InferenceClient
client = InferenceClient(api_key="")

df_val = pd.read_csv("/home/mkapadni/work/anlp_project/data/MTL_val.csv")
print(df_val.shape)
# df_val.head()

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message.content)


prompt =""" **Prompt:**  
You are a structured data extraction model. Given an input caption and additional fields separated by special tokens (`<SEP>` and `<END>`), you will extract the specified fields and return them in a JSON format. Use the following guidelines:

- Parse the fields `Caption`, `ObjRec`, `TextRec`, `ColorRec`, `Count`, and `Answer`.  
- Each field corresponds to a key in the JSON.  
- Map the text or numerical value after the colon for each field to its corresponding key in the JSON.  
- Ensure the JSON output is formatted correctly with double-quoted keys and values.

**Example Input:**  
```
Caption: A basket of red apples with a price tag reading $5.<SEP>ObjRec:1<SEP>TextRec:1<SEP>ColorRec:1<SEP>Count:1<SEP>Answer:Red<END>
```

**Expected Output:**  
```json
{
  "Caption": "A basket of red apples with a price tag reading $5.",
  "ObjRec": 1,
  "TextRec": 1,
  "ColorRec": 1,
  "Count": 1,
  "Answer": "Red"
}
```

**Task:**  
Now, for the given input, generate a JSON object following the same rules.

**Input:**  
{Input here}

**Output:**  """


preds_file = json.load(open("/home/mkapadni/work/anlp_project/src/save/vizwiz/v1_delta_new_prompt_only_skill_and_vqa/result/result_1_1334.json"))

file_names_preds = list(preds_file.keys())


def give_json_preds_from_output(input_str):
    # format input_str in the {Input here} placeholder of the prompt
    temp_prompt = prompt.replace("{Input here}", input_str)

    messages = [
            {
                "role": "user",
                "content": temp_prompt
            }
        ]

    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct", 
        messages=messages, 
        max_tokens=500
    )

    output = completion.choices[0].message.content
    # extract the json output from the completion
    # find the { and } and extract the string between them
    start = output.find("{")
    end = output.find("}")
    output = output[start:end+1]
    # format into a json
    output = json.loads(output)
    return output

preds_dict = {}
error_files = []

for file_name in tqdm(file_names_preds, total=len(file_names_preds)):
    try:
        preds= give_json_preds_from_output(preds_file[file_name][0])
        preds_dict[file_name] = preds
    except:
        error_files.append(file_name)
            
print(len(preds_dict), len(error_files))

# create a dataframe from the predictions
df_preds = pd.DataFrame(preds_dict).T
df_preds.reset_index(inplace=True)
df_preds.rename(columns={"index":"IMG"}, inplace=True)


df_preds.to_csv("/home/mkapadni/work/anlp_project/src/preds_organized/skills_vqa.csv", index=False)


