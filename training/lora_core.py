from __future__ import annotations
import json,torch,sys
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments,Trainer,DataCollatorForSeq2Seq
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from datasets import Dataset

ROOT=Path.cwd()
DATA=ROOT/"finetune_train.jsonl"
ADAPTER=ROOT/"qwen14b_lora_v2"
MODEL="Qwen/Qwen2.5-14B-Instruct"


def main():
    global MODEL, ADAPTER
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init-adapter", default=None, help="从已有 adapter 权重续训")
    args = ap.parse_args()
    MODEL = args.model
    ADAPTER = Path(args.adapter) if args.adapter else ROOT / "outputs/adapters/qwen7b_lora_v1"
    rows = [json.loads(l) for l in DATA.read_text().splitlines()]
    print(f"data: {len(rows)}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fmt(r):
        msgs = [{'role': 'user', 'content': r['user']},
                {'role': 'assistant', 'content': r['assistant']}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = tok(text, truncation=True, max_length=1024)
        return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'],
                'labels': enc['input_ids']}

    ds = Dataset.from_list(rows).map(fmt, remove_columns=['uid', 'user', 'assistant'])
    print(f"ds: {len(ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)  
    model.config.use_cache = False
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"continuing from {args.init_adapter}", flush=True)
        model.print_trainable_parameters()
    else:
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(ADAPTER), per_device_train_batch_size=4,
        gradient_accumulation_steps=8, num_train_epochs=args.epochs,
        learning_rate=args.lr, warmup_steps=50, lr_scheduler_type="cosine",
        logging_steps=50, save_strategy="no", report_to=[],
        bf16=True, gradient_checkpointing=True, dataloader_num_workers=4,
        ddp_find_unused_parameters=False, optim="adamw_torch")
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8))
    trainer.train()
    trainer.save_model(str(ADAPTER))
    tok.save_pretrained(str(ADAPTER))
    print("ADAPTER SAVED ->", ADAPTER)
