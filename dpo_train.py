from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
import torch
from pathlib import Path
# 1. Base SFT 모델 불러오기
base_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 2. LoRA 설정
peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj','v_proj']
)

# 3. LoRA 적용 모델 생성
model = get_peft_model(base_model, peft_config)
# 4. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 5. DPO용 데이터셋 로드
train_dataset = load_dataset("jinseob/patent_qa_dpo", split="train")

# 6. DPO 학습 설정
training_args = DPOConfig(
    output_dir="./dpo_output_patent_v3",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=256,
    num_train_epochs=4,
    logging_steps=1,
    learning_rate=1e-4,
    save_strategy="epoch",
    save_total_limit=4,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    max_length=1024
)

# 7. DPOTrainer 구성
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer  # 최신 trl은 tokenizer를 processing_class로 받음
)

# 8. 학습 시작
print("🔥 DPO 학습 시작!")
trainer.train()
