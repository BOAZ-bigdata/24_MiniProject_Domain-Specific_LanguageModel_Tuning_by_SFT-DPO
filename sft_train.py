from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset, load_from_disk

# 모델 설정
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# 모델 로딩 (최적화)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",        # fp16 자동 감지
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Hugging Face 업로드된 데이터셋 (ChatML 형식) -> 결합된 데이터셋 로드로 변경
dataset = load_dataset("joonkeene/QA_patent_SFT", split="train")
print(dataset)
# LoRA 설정
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj','o_proj','k_proj', 'v_proj']
)

# 학습 설정
training_args = SFTConfig(
    output_dir="./sft_output_lora_hyejung",
    per_device_train_batch_size=1,  # 사용자가 설정한 값 유지
    gradient_accumulation_steps=8,   # 사용자가 설정한 값 유지
    learning_rate=1e-4,
    num_train_epochs=6,
    bf16=True,                      # 4090은 bfloat16 지원
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=10,
    report_to="wandb", # wandb 오류 방지         # 최적화 설정 복원
    max_length=512,
)

# Trainer 구성
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
)

# 학습 시작
trainer.train()