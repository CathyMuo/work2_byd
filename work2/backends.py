"""Local inference adapters; optional dependencies imported only at runtime."""
STOP_STRINGS = ["</search>", "</answer>"]


class TokenizerMixin:
    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def initial_prompt(self, text):
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        return text


class VLLMGenerator(TokenizerMixin):
    def __init__(self, checkpoint, agent, tensor_parallel_size, memory_fraction):
        from vllm import LLM, SamplingParams
        self.sampling_class = SamplingParams
        self.model = LLM(model=str(checkpoint), tokenizer=str(checkpoint), dtype="bfloat16",
                         tensor_parallel_size=tensor_parallel_size,
                         gpu_memory_utilization=memory_fraction,
                         max_model_len=agent["max_context_tokens"], seed=agent["seed"],
                         trust_remote_code=False)
        self.tokenizer = self.model.get_tokenizer()

    def generate(self, tokens, max_tokens, seed):
        params = self.sampling_class(temperature=0.0, max_tokens=max_tokens, seed=seed,
                                     stop=STOP_STRINGS, include_stop_str_in_output=True)
        result = self.model.generate([{"prompt_token_ids": list(tokens)}], params,
                                     use_tqdm=False)[0].outputs[0]
        return {"text": result.text, "token_ids": list(result.token_ids),
                "finish_reason": result.finish_reason}

    def sample_batch(self, prompts, max_tokens, seeds, temperature, top_p=1.0):
        """One batched vLLM call; each request has its own seed (CT1 on-policy sampling)."""
        params = [self.sampling_class(temperature=temperature, top_p=top_p, max_tokens=m, seed=s,
                                      stop=STOP_STRINGS, include_stop_str_in_output=True)
                  for m, s in zip(max_tokens, seeds)]
        results = self.model.generate([{"prompt_token_ids": list(p)} for p in prompts], params, use_tqdm=False)
        return [{"text": r.outputs[0].text, "token_ids": list(r.outputs[0].token_ids),
                 "finish_reason": r.outputs[0].finish_reason} for r in results]


class HFGenerator(TokenizerMixin):
    def __init__(self, checkpoint):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
        if not torch.cuda.is_available():
            raise RuntimeError("HF backend requires a CUDA GPU for this experiment")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map="auto").eval()
        # Qwen can merge the preceding space/newline into a boundary token.
        patterns = [self.encode(prefix + stop + suffix) for stop in STOP_STRINGS
                    for prefix in ("", " ") for suffix in ("", "\n", "\n\n")]
        class StopAction(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return any(input_ids.shape[1] >= len(p)
                           and input_ids[0, -len(p):].tolist() == p for p in patterns)
        self.stop = StopAction()

    def generate(self, tokens, max_tokens, seed):
        from transformers import StoppingCriteriaList
        torch = self.torch
        torch.manual_seed(seed)
        input_ids = torch.tensor([tokens], device=self.model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False,
                max_new_tokens=max_tokens, stopping_criteria=StoppingCriteriaList([self.stop]),
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id))
        ids = output[0, len(tokens):].tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        stopped = any(stop in text for stop in STOP_STRINGS)
        return {"text": text, "token_ids": ids,
                "finish_reason": "stop" if stopped or len(ids) < max_tokens else "length"}
