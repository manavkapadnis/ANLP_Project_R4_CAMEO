# ANLP_Project_R4: CAMEO -> CAption enhanced Multi-task Optimization Framework for VQA

```
.
├── src/scripts/delta_run_3_epochs_skill_before_caption.sh      # Bash script for executing MTL baseline: skill-> caption ->vqa
├── src/scripts/delta_run_3_epochs.sh                           # Bash script for executing MTL baseline: captin-> skill ->vqa
├── src/scripts/delta_run_3_epochs_only_skill_and_vqa.sh        # Bash script for executing MTL baseline: skill->vqa
├── src/scripts/delta_run_3_epochs_only_vqa.sh                  # Bash script for executing vqa task alone
└── github_url.txt                                              # URL link to the GitHub repository
```
## Folder details: 

### src
The `src` directory contains the following components:

- **Python Scripts**: Implementation of Multi-Task Learning (MTL) baselines.
- **Bash Files**: Scripts to execute the Python implementations seamlessly.
- **Predictions**: Model-generated predictions from the baselines.
- **Evaluations**: Scripts and results for evaluating the performance of the models.

### github_url.txt
Contains the URL link to the GitHub repository for easy access to the project's source code and documentation.

## Running the Model

The `src` directory contains Python scripts that implement different Multi-Task Learning (MTL) configurations. These scripts define the task sequences for model training and evaluation.

#### Python Scripts and Configurations:
- **`delta_run_3_epochs_skill_before_caption.py`**  
  Implements the MTL baseline in the sequence: **Skill → Caption → VQA**.

- **`delta_run_3_epochs.py`**  
  Implements the MTL baseline in the sequence: **Caption → Skill → VQA**.

- **`delta_run_3_epochs_only_skill_and_vqa.py`**  
  Implements the MTL baseline with only **Skill → VQA**.

- **`delta_run_3_epochs_only_vqa.py`**  
  Implements the STL pipeline for **VQA** only.

#### How to Run
You can run these Python scripts directly or use the corresponding bash scripts from the `src/scripts` directory for convenience. The bash scripts set up the required environment and execute the respective Python scripts.

1. **Running Python Scripts Directly:**
   Navigate to the `src` directory and execute the desired script:
   ```bash
   cd src
   python3 delta_run_3_epochs_skill_before_caption.py

2. **Running through Bash files:**
   Navigate to the `src\scripts` directory and execute the desired script:
   ```bash
   cd src/scripts
   bash delta_run_3_epochs_skill_before_caption.sh

Choose the appropriate file within these folders to run the desired task or model.

3. **Modify Paths if Necessary**: Update any file paths within the code to match your local environment setup.

4. **Modify the Bash Scripts if Needed: Adjust hyperparameters or other settings before execution.

5. **Execute the Code**: Run the script or notebook to perform the desired task. This can be done directly in your preferred Python environment or Jupyter Notebook.

