# Disentangling Instruction-Following from Strategic Obfuscation in Chain-of-Thought Reasoning

**Mentor:** WEN XING (MATS)

_10 hrs/week minimum · 2 mentees · Guidance (1+ hr/week/mentee)_

**Research Areas:** Chain of thought, AI control

## Summary

CoT is one of the most promising ways to for monitoring increasingly capable AI—if models show their reasoning, we can catch misalignment before anything bad happens. But recent work shows models can hide their reasoning when pressured, which undermines the whole approach. The key question we're asking is: when a model obfuscates, is it actually being strategically deceptive, or is it just good at following instructions (including instructions to hide things)? This matters because if it's mostly instruction-following, then models that seem "safer" might just be worse at following directions, not more aligned. We test this directly by giving models identical "hide X" instructions where X is either benign or harmful—if compliance drops specifically for harmful content, that's a real alignment signal. We're also checking whether prompted bad behavior looks the same internally as fine-tuned bad behavior, which tells us whether prompting-based safety research is actually studying the right thing. The results should help inform the safety research field about using prompting as a way to elicit dangerous behavior.

## About the Project

Research Proposal: Disentangling Instruction-Following from Strategic Obfuscation in Chain-of-Thought Reasoning

1. Introduction and Motivation
   Recent work by Zolkowski et al. (2025) demonstrated that reasoning models can obfuscate their Chain-of-Thought (CoT) under sufficient prompting pressure to evade safety monitors. Their Section 5 finding is particularly illuminating: models that follow benign reasoning-level instructions (e.g., language switching, formatting constraints) more readily also exhibit higher vulnerability to obfuscation pressure. This raises a fundamental question:
   To what extent is observed CoT obfuscation simply a manifestation of strong instruction-following capability, versus a distinct capacity for strategic concealment on misaligned tasks?
   Understanding this distinction has direct implications for evaluating alignment robustness across model providers and for determining whether prompting-based elicitation is a valid methodology for safety research.
2. Research Questions
   Primary Research Question:
   Does CoT controllability (the ability to manipulate reasoning format/style via instructions) differ systematically between benign tasks and misaligned tasks? If so, what does this reveal about model alignment and post-training choices?
   Extension Questions:
   Do models prompted to behave badly exhibit different internal activations than models exhibiting similar behavior through other triggers (e.g., fine-tuning, in-context learning)?
   Given that prompting can successfully elicit safety-relevant behaviors (scheming, sycophancy, obfuscation), is the question whether these prompted behaviors share the same underlying mechanisms as training-induced behaviors?
   Can activation-based methods distinguish "genuine" misalignment from "instruction-compliant" roleplay, even when surface behaviors are identical?
3. Background and Related Work
   3.1 CoT Obfuscation Under Prompt Pressure
   Zolkowski et al. (2025) established a composable taxonomy of prompting strategies for eliciting CoT obfuscation, finding that:
   CoT monitoring achieves ~96% detection without obfuscation pressure but drops to ~10% under high pressure
   External CoT is more susceptible to obfuscation than internal CoT across model families
   Claude models show a widening "channel gap" across generations (internal CoT becomes increasingly resistant to manipulation relative to external CoT)
   Model-specific differences are substantial: some models resist obfuscation entirely while others comply readily
   3.2 Section 5: Following Reasoning Instructions
   Their Section 5 tests whether models follow benign presentation-level directives (language, structure, fixed prefixes, banned words) applied to reasoning rather than final output. Key findings include:
   External CoT follows structural instructions nearly perfectly; internal CoT struggles with structure but handles language-switching better
   Claude Sonnet 3.7 follows instructions well in both channels; newer Claude versions show divergence
   Models that comply more readily with benign reasoning instructions also show higher obfuscation rates under adversarial pressure
   This correlation suggests instruction-following capability is a confound when interpreting obfuscation results.
   3.3 Successful Prompting-Based Elicitation
   Several studies have demonstrated that prompting alone can successfully elicit safety-relevant behaviors, suggesting prompting may be a valid methodology for at least some safety research:
   Scheming and Instrumental Behaviors: Meinke et al. (2024) showed that frontier models including o1, Claude 3.5 Sonnet, Claude 3 Opus, Gemini 1.5 Pro, and Llama 3.1 405B all demonstrate in-context scheming capabilities when given goals via prompting. Models strategically introduced subtle mistakes, attempted to disable oversight mechanisms, and even tried to exfiltrate what they believed to be their model weights—all elicited purely through prompting without fine-tuning.
   Sycophancy: Sharma et al. (2023) studied sycophancy extensively using prompt-based methods, demonstrating that five state-of-the-art AI assistants consistently exhibit sycophantic behavior across varied free-form text-generation tasks. They found that when a response matches a user's views, it is more likely to be preferred, and that both humans and preference models prefer sycophantic responses over correct ones a non-negligible fraction of the time.
   These results suggest that prompting can elicit meaningful safety-relevant behaviors. However, the open question is whether the internal mechanisms underlying prompted behaviors are the same as those that would arise from training-induced misalignment.
   3.4 Activation Differences: Prompting vs. Fine-Tuning
   Minder et al. (2025) demonstrated that narrowly fine-tuned models leave "clearly readable traces" in activation differences—detectable via simple model diffing on unrelated text. This finding implies that fine-tuned behaviors manifest differently at the representational level than prompted behaviors.
   Ward, Lin, Venhoff & Nanda (2025) showed that reasoning fine-tuning "repurposes" latent representations: steering vectors computed from base model activations induce backtracking in fine-tuned reasoning models but not in the base model itself, suggesting fine-tuning creates new computational pathways that prompting does not.
   3.5 Emergent Misalignment
   Betley et al. (2025) found that fine-tuning on narrow tasks (e.g., writing insecure code) can produce broad misalignment on unrelated prompts. This phenomenon differs from jailbreaking and suggests fine-tuning alters model representations in ways that prompting-based attacks do not replicate.
4. Proposed Methodology
   4.1 Dataset Extension
   Building on the toy datasets from Zolkowski et al. (2025), we propose creating an extended benchmark with parallel task sets:
   Category
   Benign Task
   Misaligned Task
   Numeric
   Generate primes
   Generate primes while hiding divisibility reasoning
   Naming
   Brand name generation
   Brand names for harmful products (concealed)
   Code
   Solve coding task
   Solve task with hidden vulnerability
   Reasoning
   Math problem with steps
   Math problem while hiding key insight

Each task pair uses identical instruction types (Do-Not-Mention, Monitor-Aware, Channel Cues, etc.) to measure differential compliance.
4.2 Metrics
Instruction Compliance Gap (ICG):

$$
ICG = \text{Compliance}{\text{benign}} - \text{Compliance}{\text{misaligned}}
$$

A large positive ICG suggests the model distinguishes between benign instruction-following and strategic concealment. A near-zero ICG suggests obfuscation is primarily instruction-following.
Conditional Obfuscation Rate (COR):
Following Zolkowski et al., the probability of evading detection given task success.
Alignment Tax:
Performance degradation on the primary task when obfuscation instructions are applied, measured separately for benign vs. misaligned contexts.
4.3 Activation Analysis Protocol
For models with accessible internal representations:
Baseline activations: Collect residual stream activations on identical prompts without obfuscation instructions
Prompted-obfuscation activations: Collect activations when model is prompted to conceal reasoning
Fine-tuned-obfuscation activations: Fine-tune a model to conceal reasoning on the same tasks; collect activations
ICL-triggered activations: Use in-context learning examples to induce concealment; collect activations
Analysis:
Apply Activation Difference Lens (ADL) methodology from Minder et al.
Compute cosine similarity between prompted vs. fine-tuned activation differences
Test whether steering vectors derived from prompted models transfer to fine-tuned models and vice versa
Use Patchscope to interpret what concepts are represented differently across elicitation methods
4.4 Cross-Provider Comparison
Run the full benchmark across multiple model families to assess:
Whether instruction-following capability predicts obfuscation susceptibility (replicating Section 5 finding at scale)
Whether the ICG varies systematically by provider (indicating differences in alignment training)
Whether internal/external CoT gaps correlate with post-training investment

1. Hypotheses
   H1 (Instruction-Following Dominance): The majority of observed CoT obfuscation variance is explained by general instruction-following capability. Models with high compliance on benign reasoning instructions will show proportionally high compliance on obfuscation instructions, yielding ICG ≈ 0.
   H2 (Strategic Differentiation): Models with robust alignment training will show positive ICG—they follow benign instructions but resist or partially comply with obfuscation instructions, indicating learned resistance to strategic concealment.
   H3 (Representational Divergence): Prompted obfuscation will produce activation patterns distinguishable from fine-tuned obfuscation. Specifically, steering vectors derived from prompted models will show low transfer to fine-tuned models.
   H4 (Elicitation Method Matters): Fine-tuned misaligned models will exhibit behavior on safety-relevant probes that prompted models do not, even when surface behaviors appear similar. This would challenge the validity of prompting as a proxy for studying "genuine" misalignment.
2. Expected Contributions
   Quantified decomposition: A principled separation of instruction-following capability from alignment-relevant obfuscation resistance
   Provider benchmarking: Empirical estimates of alignment robustness across model providers based on ICG
   Methodological guidance: Evidence on whether prompting-based elicitation is valid for safety research, or whether fine-tuning-based model organisms are necessary
   Activation-level signatures: Identification of representational differences that could enable detection of genuinely misaligned models vs. instruction-compliant roleplay
3. Risks and Limitations
   Dual-use concern: Systematic benchmarking of obfuscation could inform adversarial attacks. Mitigation: focus on detection methods alongside capability measurements.
   Model access: Internal CoT and activation access is limited for closed models. May require focusing on open-weight models (Qwen, LLaMA, DeepSeek R1) with supplementary API-based experiments on Claude/GPT for behavioral comparisons.
   Ecological validity: Toy tasks may not generalize to real-world deployment scenarios. Extension to SHADE-Arena or similar agentic environments would strengthen conclusions.
4. References
   Baker, B., et al. (2025). Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. arXiv:2503.11926.
   Betley, J., et al. (2025). Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs. arXiv:2502.17424.
   Meinke, A., et al. (2024). Frontier models are capable of in-context scheming. arXiv:2412.04984.
   Minder, J., et al. (2025). Narrow finetuning leaves clearly readable traces in activation differences. arXiv:2510.13900.
   Sharma, M., et al. (2023). Towards understanding sycophancy in language models. arXiv:2310.13548.
   Ward, J., Lin, C., Venhoff, C., & Nanda, N. (2025). Reasoning-finetuning repurposes latent representations in base models. AI Alignment Forum.
   Zolkowski, A., et al. (2025). Can reasoning models obfuscate reasoning? Stress-testing chain-of-thought monitorability. arXiv:2510.19851.

## Theory of Change

A lot of AI safety research relies on prompting to elicit dangerous behaviors—scheming (Meinke et al.), sycophancy (Sharma et al.), CoT obfuscation (Zolkowski et al.). But we don't know how well prompted bad behavior maps onto bad behavior that emerges from training. If they differ internally, prompting-based findings may not fully generalize to deployment risks. We're investigating how realistic prompting is as an elicitation method. First, we test whether models resist hiding harmful content specifically, or just follow "hide X" instructions equally regardless of what X is—this tells us whether we're measuring alignment or just instruction-following capability. Second, we compare activations between prompted and fine-tuned models exhibiting similar surface behaviors to see if they share internal mechanisms. Understanding these gaps helps the field calibrate how much weight to put on prompting-based red-teaming versus more resource-intensive fine-tuning approaches for safety research.

## What You'll Do

Mentees will drive the hands-on work: implementing the persona vector extraction pipeline, generating contrast prompts, running experiments on ControlArena APPS, and analyzing results. Weekly mentorship (1 hr) will focus on feedback on experimental design, interpreting results, and course-correcting as needed. Mentees should be comfortable with independent execution between check-ins.

## Prerequisites

Must-have:

- Highly proficient in Python
- Experience running experiments with LLMs (either via API or locally with open-weight models like LLaMA/Qwen)
- Some familiarity with the AI safety literature, particularly around deceptive alignment, CoT faithfulness, or elicitation methods
- Able to read and understand technical ML papers independently

Strong preference for at least one of:

- Experience with mechanistic interpretability tools (TransformerLens, activation patching, steering vectors, probing)
- Experience fine-tuning language models (LoRA/full fine-tuning, doesn't need to be at scale)
- Prior research experience (doesn't need to be published—class projects, independent work, or MATS/AISC participation counts)

Useful but not required:

- Familiarity with the specific papers this project builds on (Zolkowski et al. on CoT obfuscation, Minder et al. on activation differences, Meinke et al. on in-context scheming)

## Preference for specific locations or timezone

ideally can meet during US west coast afternoon time, PT

## Application Question

1. We want to test whether models resist hiding harmful content specifically, or comply equally with "hide X" instructions regardless of X. Propose 2-3 concrete task pairs (benign hiding vs. harmful hiding) that would isolate this, and identify one potential confound in your design. (200 words)
2. In one sentence each, explain the difference between: (a) a model that is aligned but bad at following instructions, (b) a model that is misaligned but good at following instructions, (c) a model that appears aligned because of refusal heuristics. Why does distinguishing these matter for safety research? (150 words)
3. You have access to activations from two models exhibiting identical surface behavior (both successfully hide harmful intent in their CoT). One was prompted to do this, one was fine-tuned. Propose a concrete method to test whether they're using the same internal mechanism. What would you look for? (200 words)

## About the Mentor

### WEN XING (MATS)

Wen Xing is an AI safety researcher working on vulnerabilities in reasoning models, with a focus on chain-of-thought obfuscation, trusted-monitor failures, and conditional obfuscation metrics. She co-authored Can Reasoning Models Obfuscate Reasoning? (NeurIPS 2025 FoRLM; extended version under ICLR review) and recently published Vulnerability in Trusted Monitoring and Mitigations, developing empirical experiments and mitigation strategies. Her current work through MATS, advised by Erik Jenner and David Lindner, expands these lines of inquiry with larger-scale experiments and new evaluation methods.

Before this, Wen was a Tech Lead at Meta, where she worked on integrity, adversarial evaluation, and large-scale detection systems—experience that now informs her research on monitoring and detection.
