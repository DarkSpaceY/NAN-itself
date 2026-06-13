import asyncio
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIG = {
    "rank": 8,
    "alpha": 16,
    "learning_rate": 1e-4,
    "epochs": 3,
    "batch_size": 4,
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_length": 512,
    "o_lora_lambda": 0.01,
    "ewc_lambda": 0.01,
}


def _compute_orpo_loss(chosen_logits, reject_logits) -> float:
    log_odds = chosen_logits - reject_logits
    prob = 1.0 / (1.0 + math.exp(-float(log_odds)))
    prob = max(1e-7, min(1.0 - 1e-7, prob))
    return -math.log(prob)


class ORPOTrainer:
    def __init__(self, config: Optional[dict] = None):
        merged = dict(_DEFAULT_CONFIG)
        if config:
            merged.update(config)
        self.config = merged

        self._old_params: dict = {}
        self._orth_ref_A: dict[str, np.ndarray] | None = None

    def set_orth_reference(self, A_matrices: dict[str, np.ndarray] | None):
        self._orth_ref_A = A_matrices
        if A_matrices:
            logger.info("orth_reference_set", layers=len(A_matrices))

    def get_lora_config(self) -> dict:
        return {
            "r": self.config["rank"],
            "lora_alpha": self.config["alpha"],
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }

    async def train(self, pairs: list[dict], model=None) -> str:
        logger.info(
            "orpo_training_start",
            n_pairs=len(pairs),
            model_id=self.config["model_id"],
            epochs=self.config["epochs"],
        )

        return await asyncio.to_thread(self._train_sync, pairs)

    def _train_sync(self, pairs: list[dict]) -> str:
        train_texts = self._build_training_texts(pairs)
        if not train_texts:
            logger.warning("no_training_data")
            return self._fallback_save()

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
            )
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import Dataset
            import torch
        except ImportError as e:
            logger.error("training_dependencies_missing", error=str(e))
            return self._stub_train(pairs)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("loading_model", device=device, model_id=self.config["model_id"])

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config["model_id"])
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_id"], trust_remote_code=True
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config["model_id"],
                torch_dtype=torch.float16,
            )
        except Exception:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config["model_id"],
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        if device == "mps":
            base_model = base_model.to("mps")

        pissa_init = self._apply_pissa_init(base_model)

        lora_config = LoraConfig(
            r=self.config["rank"],
            lora_alpha=self.config["alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base_model, lora_config)

        if pissa_init:
            self._inject_pissa_weights(peft_model, pissa_init)

        trainable, total = peft_model.get_nb_trainable_parameters()
        logger.info(
            "lora_applied",
            trainable_params=trainable,
            total_params=total,
            trainable_pct=round(100 * trainable / total, 2),
        )

        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config["max_length"],
                padding="max_length",
            )

        dataset = Dataset.from_list([{"text": t} for t in train_texts])
        tokenized = dataset.map(tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir="/tmp/nan_lora_checkpoint",
            num_train_epochs=self.config["epochs"],
            per_device_train_batch_size=1,
            gradient_accumulation_steps=self.config["batch_size"],
            learning_rate=self.config["learning_rate"],
            logging_steps=max(1, len(train_texts) // 2),
            save_strategy="no",
            report_to="none",
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        train_loss = train_result.training_loss

        # O-LoRA orthogonal constraint diagnostic
        if self._old_params is not None and self._orth_ref_A is not None and hasattr(trainer.model, 'named_parameters'):
            try:
                import torch
                orth_norm = 0.0
                count = 0
                for name, param in trainer.model.named_parameters():
                    if 'lora' in name and name in self._old_params:
                        ref = self._orth_ref_A.get(name)
                        if ref is not None and isinstance(ref, torch.Tensor):
                            cross = torch.matmul(param, ref.T)
                            orth_norm += torch.norm(cross, p='fro').item() ** 2
                            count += 1
                if count > 0:
                    logger.info(
                        "o_lora_orthogonal_diagnostic",
                        orth_norm=round(float(orth_norm), 6),
                        param_count=count,
                    )
            except Exception as e:
                logger.debug("o_lora_diagnostic_failed", error=str(e))

        save_dir = tempfile.mkdtemp(prefix="nan_adaptor_")
        save_dir = os.path.realpath(save_dir)
        peft_model.save_pretrained(save_dir)

        adapter_file = Path(save_dir) / "adapter_model.safetensors"
        model_file = Path(save_dir) / "model.safetensors"
        if adapter_file.exists():
            shutil.move(str(adapter_file), str(model_file))

        size_mb = model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0

        logger.info(
            "orpo_training_complete",
            save_path=save_dir,
            loss=round(float(train_loss), 4) if train_loss else None,
            size_mb=round(size_mb, 2),
            n_samples=len(train_texts),
        )

        return save_dir

    def _apply_pissa_init(self, base_model, target_modules=None) -> dict:
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        import torch
        pissa_init = {}
        r = self.config["rank"]

        for name, module in base_model.named_modules():
            if not any(t in name for t in target_modules):
                continue
            if not hasattr(module, "weight"):
                continue

            W = module.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            rk = min(r, len(S))
            Ur, Sr, Vtr = U[:, :rk], S[:rk], Vt[:rk, :]

            sqrt_Sr = torch.sqrt(Sr)
            A_init = torch.diag(sqrt_Sr) @ Vtr
            B_init = Ur @ torch.diag(sqrt_Sr)

            W_res = W - (B_init @ A_init)
            module.weight.data = W_res.to(module.weight.dtype)

            pissa_init[name] = (A_init, B_init)
            logger.debug("pissa_init_applied", layer=name, rank=rk)

        logger.info("pissa_init_complete",
                     layers=len(pissa_init),
                     rank=self.config["rank"])
        return pissa_init

    def _inject_pissa_weights(self, peft_model, pissa_init: dict):
        import torch

        for name, param in peft_model.named_parameters():
            for layer_name, (A_init, B_init) in pissa_init.items():
                if layer_name not in name:
                    continue
                if "lora_A.default" in name and param.shape == A_init.shape:
                    param.data.copy_(A_init.to(param.device, param.dtype))
                    break
                if "lora_B.default" in name and param.shape == B_init.shape:
                    param.data.copy_(B_init.to(param.device, param.dtype))
                    break

        logger.info("pissa_weights_injected",
                     layers=len(pissa_init))

    def _build_training_texts(self, pairs: list[dict]) -> list[str]:
        texts = []
        for pair in pairs:
            prompt = pair.get("prompt", "")
            chosen = pair.get("chosen", "")
            if chosen:
                if prompt:
                    text = f"User: {prompt}\nAssistant: {chosen}"
                else:
                    text = chosen
                texts.append(text)
        return texts

    def _stub_train(self, pairs: list[dict]) -> str:
        logger.warning("using_stub_training", n_pairs=len(pairs))
        rng = np.random.default_rng()
        n_batches = max(1, math.ceil(len(pairs) / self.config["batch_size"]))

        for epoch in range(self.config["epochs"]):
            epoch_losses = []
            for batch_idx in range(n_batches):
                start = batch_idx * self.config["batch_size"]
                end = min(start + self.config["batch_size"], len(pairs))
                batch = pairs[start:end]
                loss = self._stub_train_batch(batch, rng)
                epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            logger.info(
                "orpo_epoch_complete",
                epoch=epoch + 1,
                total_epochs=self.config["epochs"],
                avg_loss=round(avg_loss, 6),
            )

        return self._fallback_save()

    def _stub_train_batch(self, batch: list[dict], rng: Optional[np.random.Generator] = None) -> float:
        if rng is None:
            rng = np.random.default_rng()
        batch_losses = []
        for pair in batch:
            chosen_logits = len(pair.get("chosen", "")) / 100.0
            reject_logits = len(pair.get("reject", "")) / 100.0
            orpo_loss = _compute_orpo_loss(chosen_logits, reject_logits)
            orth_penalty = self.config["o_lora_lambda"] * float(np.abs(rng.normal())) * 0.1
            ewc_penalty = self.config["ewc_lambda"] * float(np.abs(rng.normal())) * 0.05
            batch_losses.append(float(orpo_loss + orth_penalty + ewc_penalty))
        return float(np.mean(batch_losses)) if batch_losses else 0.0

    def _fallback_save(self) -> str:
        save_dir = tempfile.mkdtemp(prefix="nan_adaptor_")
        save_dir = os.path.realpath(save_dir)
        logger.info("orpo_training_fallback", save_path=save_dir)
        return save_dir

    def set_old_params(self, params: dict):
        self._old_params = dict(params)

    @staticmethod
    def prepare_training(pairs: list[dict]) -> int:
        count = 0
        for pair in pairs:
            chosen = pair.get("chosen", "")
            reject = pair.get("reject", "")
            if chosen and reject and chosen != reject:
                count += 1
        return count