This study adopts a sequential latent-state modelling framework to capture the evolving phys

iological dynamics underlying the MC. Rather than treating daily observations as independent

samples, the proposed model explicitly represents the cycle as a temporally structured pro

cess driven by gradual endocrine transitions. The core objective is to infer a latent cycle state

that evolves with daily physiological inputs and supports both ovulation-related inference and

menstruation forecasting.

Let {x1, x2, . . . , xt} denote the sequence of daily multimodal feature vectors constructed through

physiological aggregation and within-individual normalisation. A recurrent cycle state tracker

updates a hidden state ht recursively according to

ht = f(xt
, ht−1),

(3.3)

where f(·) denotes a recurrent update function. In this work, a Gated Recurrent Unit (GRU) is

used as the primary architecture, with a Long Short-Term Memory (LSTM) network trained as

a structural comparator under identical protocols. Recurrent models are selected because they

provide an explicit mechanism for modelling gradual state evolution and temporal dependency

while remaining parameter-efficient under limited sample sizes. More expressive architectures

such as Transformers or recent state-space models (e.g., Mamba) were not adopted, as their

advantages typically emerge in large-scale regimes with long sequences and abundant training

data. Under the present small-cohort setting, recurrent models offer a more suitable inductive

bias and improved sample efficiency, while also ensuring computational tractability.

Two task-specific prediction tasks operate on the shared latent state ht
, forming a sequential

multi-task learning framework. The primary task is menstruation forecasting, formulated as a

time-to-event regression problem. For each day t, the model predicts the remaining number of

days until the onset of the next menstrual bleed:

yˆt = Wmht + bm.

(3.4)

This task benefits from dense supervision, as menstruation onset dates are available for all

complete cycles. The corresponding loss is computed for all time steps with observable ground

truth and is masked for truncated cycle boundaries where future onset information is unavailable.

The auxiliary task is ovulation discrimination (not prediction): given the sequence x1, . . . , xt

up to and including day t, the model outputs the probability that day t is the ovulation day. This is

an online discriminative problem—P(today is ovulation day | physiological signals so far)—not a

long-horizon forecast of when ovulation will occur. Ovulation is a single-point event; in wearable-

only settings the discriminative signal is inherently weak: LH surge (pre-ovulation) is not observed

in the input; temperature rise is post-ovulation (retrospective); HRV changes are typically phase-

level rather than day-precise. Thus the mapping xt → ovulation_t is only weakly supported by the

available features. Formally, given the hidden state ht (which summarises x1, . . . , xt), the ovulation

head estimates

p(day t is ovulation | h_t) = σ(Wovht + bov),

(3.5)

where σ(·) denotes the sigmoid function. Ovulation labels are derived from endocrine patterns

based on luteinising hormone (LH) and oestrogen measurements and are available only for a

subset of cycles and time steps. Importantly, hormonal measurements are used exclusively for

annotation and evaluation purposes and are not included in the model input features, ensuring

a wearable-only sensing setting and preventing data leakage.

To prevent noisy or unfounded supervision, the auxiliary loss is applied strictly under a masking

scheme:

Lovulation =
X

t∈Tov

ℓbce p
(
t
ov)
, ot

,

(3.6)

where Tov denotes time steps with verified ovulation annotations. For all other time steps, the

auxiliary loss is masked to zero and does not contribute to gradient updates. This auxiliary

task therefore operates under a sparse supervision regime and is not expected to dominate

optimisation. Instead, it provides weak but structured guidance that encourages the latent

state ht to encode physiologically meaningful transitions associated with ovulation.

The total training objective is given by

L = Lmenses + λLovulation,

(3.7)

where λ controls the influence of the auxiliary task. The ovulation task is not introduced as an

additional prediction target for its own sake, but as a source of inductive bias. By constraining

the hypothesis space of the recurrent backbone, the auxiliary supervision mitigates overfitting

in a small-sample regime while preserving flexibility for the primary forecasting objective.

A two-stage training protocol is employed to balance auxiliary supervision and downstream

forecasting performance. In the first stage, the recurrent backbone and both tasks are jointly

optimised under the combined loss. In the second stage, the menstruation task is fine-tuned

independently, while the recurrent backbone is either frozen or updated under a reduced learning

rate. This design reflects the recognition that auxiliary supervision is sparse and potentially

noisy, and that excessive coupling may interfere with the optimisation of the primary task.

Alternative modelling paradigms were considered. Purely rule-based approaches, such as pre

dicting menstruation by adding a fixed or probabilistic luteal phase duration to an estimated

ovulation day, provide interpretability but are inherently rigid and unable to capture tran

sient physiological perturbations reflected in wearable signals. Conversely, purely statistical or

Bayesian cycle-length models effectively capture historical regularities but disregard contempo

raneous physiological evidence. The proposed latent-state framework occupies an intermediate

position: it preserves temporal structure and physiological grounding while enabling data-driven

adaptation under sparse and heterogeneous physiological observations.

20Figure 3.1: Overall structure of the proposed sequential multi-task latent cycle state

model with task-specific prediction heads

——— Design note (Ovulation head role) ———

• Ovulation head = classification/probability discriminative head, not a predictive head. Given x₁, x₂, …, x_t, it outputs P(today is ovulation day | physiological signals up to today).
• Problem form: online discrimination, not long-horizon prediction. Ovulation is a single-point event; wearable signals are lagged (temperature rises after ovulation; no LH input).
• Physiological facts: LH peak precedes ovulation (not in input); temperature rise is post-ovulation; HRV is phase-level. Thus in wearable-only settings, the mapping x_t → ovulation_t is very weak; the ovulation head can only provide weak inductive bias; the main task remains menses prediction.
