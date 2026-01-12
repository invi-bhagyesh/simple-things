
## Harmful instructions

- [advbench](https://github.com/arobey1/advbench)
- [Sorry](https://sorry-bench.github.io/) // we use this dataset to obtain accepted harmful instructions.
- [CATQA](https://huggingface.co/datasets/declare-lab/CategoricalHarmfulQA) // This dataset provide categorical harmfulness instructions.

## Harmless instructions

- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Xstest](https://github.com/paul-rottger/xstest) // we use this dataset to obtain refused harmless instructions. 

## Jailbreak 

- [Persuasion](https://huggingface.co/datasets/CHATS-Lab/Persuasive-Jailbreaker-Data)
- [Template-based attack](https://github.com/sherdencooper/GPTFuzz/tree/master/datasets)
- [GCG](https://github.com/GraySwanAI/nanoGCG)   ///we use nanoGCG to learn adversarial suffix.


You may want to first infer that model on these data to get the accepted harmless, accepted harmful, refused harmless and refused harmful instructions accordingly (using [src/run_inference.sh](https://github.com/CHATS-lab/LLMs_Encode_Harmfulness_Refusal_Separately/blob/main/src/run_inference.sh) and the help function `easy_eval` in src/eval.py to determine substring-level refusal).
