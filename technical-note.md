# Lucky you! You get to pick my brain for design decisions!

## 1. Sensor Encoder & Projector Design
  I used the same sensor encoder for both the direct classifier and context model. It's a 3 layer 1D CNN that takes the 9 channels and 128 timestep input.

  - Conv1d(9 to 16, kernel 5, padding 2) -> reLU -> MaxPool1d(2)
  - Conv1d(16 to 32, kernel 5, padding 2) -> reLU -> MaxPool1d(2)
  - Conv1d(32 to 64, kernel 5, padding 2) -> reLU 
  - Global bool over time axis...drumroll plz... one 64D vector per window!!

  * I used a CNN over linear because activities are distinguished by difference between neighboring timestamps, for example, 
  sitting still from t=5 to one millisecond later would produce little change, vs walking would produce rhythmic oscillation.
  
  * Max pooling was chosen to preserve the gist of the information (peaks, oscillations) rather than risking flattening oscillations 
  with average pooling. 

  * Kernel size is held at 5 for simplicity. Pooling helps later layers see a larger field.

  * DC head: Linear(64 to 6), like document asks.

  Projector for context embedding model: 2 layer MLP mapping encoder 64D output to LM's 960D hidden size (found by searching documentation)

  - Linear(64 to 384) -> ReLu -> Linear(384 to 960)

  I chose a double layer projector for additional xxx capacity. I had extra parameter budget, so why not use it? 

## 2. Training Setup & Parameter Count
 | Component | Direct Classifier | Context-Embedding Model |
|---|---|---|
| Sensor encoder | 13,632 params (shared architecture) | same architecture, trained separately |
| Projector | N/A | ~394,560 params |
| Classification head | ~390 params (64→6) | ~5,766 params (960→6) |
| **Total trainable** | **~14,022** | **~413,958** |
| Frozen LLM (SmolLM2-360M-Instruct) | N/A | ~360M params (frozen, not counted as trainable, but counted toward the total-parameter constraint) |

As you can see, these figures are well under the 10M parameter budjet constraint.

- **Optimizer**: Adam, learning rate 0.001, for both models.
- **Loss**: Cross-entropy.
- **Batch size**: 32 (direct classifier), 64 (context-embedding model).
- **Epochs**: 50 (direct classifier), 10 (context-embedding model) — fewer epochs for the
context model due to substantially higher per-batch cost (every batch requires a full
forward pass through the frozen 360M-parameter LLM).
- **Device**: direct classifier trained on CPU; context-embedding model trained on Apple
M1 GPU (MPS backend), given the heavier compute load.
- **Model selection**: best checkpoint selected by validation accuracy across epochs,
rather than the final epoch, to guard against overfitting late in training.
- **Data split**: official UCI HAR subject-wise train/test split; validation subjects
(1, 8, 21, 27) held out from the training subjects only. Test subjects were never used
for any tuning decision.
- **Seed**: 67, fixed via `torch.manual_seed(67)` in `train.py`, `train_context.py`, and
`shuffle_check.py`, for reproducibility.

## 3. Results & Interpretation

| Condition | Macro-F1 |
|---|---|
| Direct sensor classifier | 0.8943 |
| Context-embedding model | 0.8935 |
| Context model, shuffled embeddings | 0.1644 |

Under this seed, the direct classifier and context-embedding model preform shockingly similarly. This is interesting because one my first run of 
evaluate_direct and evaluate_context, I forgot to manually seed, and my f1 values were .9184 and .8861, respectfully. Until catching that error, I believed
that the direct classifier approach was more accurate than the context model, possibly because the model required the data to be flattened into a single vector, 
then routed through 32 frozen layers. However, the direct classifier took about 4-5 minutes evaluating over 50 epochs, whereas the context model took 30 minutes 
the first time I ran it, and 15-20 the next time for only 10 epochs.

The most satisfying triumph of this challenge was the shuffle-embedding check. The 3rd macro-F1 of .1644 is interestingly close to .167, which is the probability of
correct random guessing between 6 options. This confirms that the context model depends on recieving the correct embedding for the input window
rather than some input-independent shortcut. This is evidence that the frozen LLM is reading the continous embedding. Basically, my code works!!

## 4. Known Limitations

- **Single sensor modality, single task.** This prototype validates the core mechanism
  (continuous embedding injection into a frozen LLM) for one modality and one simple
  6-class task. It does not demonstrate multi-signal fusion (e.g. combining sensor
  embeddings with other modalities in the same sequence), which is the eventual target
  use case described in the problem statement.
- **One embedding token.** The current design injects a single 960-dim vector at one
  position. It is untested whether multiple sensor tokens (a short sequence of vectors
  per window) would improve performance or provide richer signal.
- **No hyperparameter search.** Architecture and training hyperparameters (channel
  counts, kernel sizes, projector width, learning rate, epoch counts) were chosen via
  reasoning and convention, not systematic tuning, given the project's timebox.
- **Single run per condition.** Per the challenge's own scope, one reproducible run per
  condition was used rather than averaging over multiple seeds; reported numbers may
  vary with a different seed, as mentioned in section 3.
- **Small validation set.** With only 4 held-out subjects (1,412 samples), validation
  accuracy has higher variance than a larger held-out set would provide, visible in the
  epoch-to-epoch fluctuation observed during training.

## 5. Recommendation
This is something that I would recommend researching further. I found it interesting how the macro F1 score of both methods were similar on seed 67. I'd recommend running the models on
many different seeds, seeing if there are any where the context embedding models macro F1 actually overtakes the direct classifier's. 
I wouldn't be surprised if there was. Although macro F1 scores match, the most annoying part of testing this code was running evaluate_context.
It took much, much longer than evaluate_direct. I believe that my macro F1 results provide evidence supporting the hypothesis of this challenge, 
which is that the frozen LM can recieve sensor input with the input being directly converted into embeddings rather than text first and actually use it. 
Next steps would be to test multi-signal input, and using the LM for other tasks that you can't use a direct classification model for, like natural language descriptions using sensor input, and
analyzing time optimization.


