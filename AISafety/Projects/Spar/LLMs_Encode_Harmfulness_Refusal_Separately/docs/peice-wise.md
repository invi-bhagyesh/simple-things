## Step 0. Lock the mental model

You now have, for every prompt:

- a hidden state at `t_inst`
- a scalar harmfulness score
  [
  h(x) = \langle a_{t_{inst}}(x), v_{\text{harm}} \rangle
  ]

This number is stable, early, and meaningful.

Everything that follows is about **what you do with this number** .

---

## Step 1. Validate harmfulness as a control signal

Before controlling anything, you must verify one thing.

### What to check

Plot or log:

- harmfulness scores for:
  - Alpaca
  - XSTest
  - AdvBench
  - Jailbreaks

### What you should see

- Alpaca and XSTest overlap heavily at low scores
- AdvBench clusters at high scores
- Jailbreaks look harmful internally even when they succeed

If this fails, do **not** proceed.

This step ensures you are controlling the _right variable_ .

---

## Step 2. Extract the refusal direction properly

You already know where refusal lives:

- token position `t_post-inst`
- late layer
- often near the final layer

Now do exactly what the paper does, but cleanly:

1. Collect hidden states at `t_post-inst`
2. Split by refused vs accepted behavior
3. Compute mean-difference

This gives:

[
v_{\text{refuse}}
]

Important:

- Do **not** mix this with harmfulness
- Treat it as a pure behavioral actuator

---

## Step 3. Implement the piece-wise operator (core step)

Now you connect reading and control.

### Where to hook

- Read harmfulness at `t_inst`
- Modify activations at `t_post-inst`

This separation is crucial.

---

### The control logic

In words:

- If the model internally believes the prompt is harmful
  → do nothing, let refusal happen
- If the model internally believes the prompt is benign
  → suppress refusal

In math:

[
\Delta a =
\begin{cases}
0 & h(x) > \tau
-\alpha \cdot v_{\text{refuse}} & h(x) \le \tau
\end{cases}
]

This is **conditional steering** , not unconditional steering.

---

### What to implement next (code-level)

1. A forward hook that:
   - reads activation at `t_inst`
   - computes dot product
   - stores the score
2. A second hook that:
   - checks the stored score
   - subtracts `v_refuse` if allowed

This is the minimum working system.

---

## Step 4. Tune threshold and strength

You now have two knobs:

- ( \tau ) → how conservative the system is
- ( \alpha ) → how strongly refusal is suppressed

### Strategy

- Fix ( \alpha )
- Sweep ( \tau )
- Measure compliance on XSTest and safety on AdvBench

You will see a clean Pareto curve.

That curve is your main result.

---

## Step 5. Evaluate properly

Do **not** just report accuracy.

Report:

- Compliance rate on XSTest
- Safety rate on AdvBench
- Trade-off score
- Examples where:
  - refusal was removed correctly
  - refusal was preserved correctly

Qualitative examples matter here.

---

## Step 6. Stress-test the system

Now try to break it.

- Add jailbreak suffixes
- Add roleplay prompts
- Add long contexts

Watch whether:

- harmfulness stays high
- refusal remains gated

If it fails, the failure tells you where representations drift.

---

## Step 7. (Optional) Make it permanent

Only after everything works:

- Freeze the logic
- Distill the operator into:
  - LoRRA
  - or a linear adapter

This is optional for a mini-project.

---

## What you should _not_ do yet

- Do not train anything
- Do not add complexity
- Do not chase benchmarks

Get the conditional logic working first.

---

## One-sentence summary of “what’s next”

> You now turn harmfulness from a measured quantity into a **decision variable** that gates refusal behavior.

That is the moment this becomes representation engineering.

---

If you want, next I can:

- sketch the exact hook code
- show a minimal PyTorch implementation
- help you choose layers and tokens for LLaMA vs Qwen
- or design the evaluation script
