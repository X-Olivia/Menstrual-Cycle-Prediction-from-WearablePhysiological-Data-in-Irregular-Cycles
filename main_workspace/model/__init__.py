"""Sequence latent-state multi-task model (menses prediction + ovulation inference)."""
from .config import (
    WORKSPACE,
    CYCLE_CSV,
    FULL_CSV,
    FEATURE_COLS,
    INPUT_DIM,
    HIDDEN_SIZE,
    RNN_TYPE,
    LAMBDA_OV,
)
from .dataset import prepare_all_sequences, CycleSequenceDataset, collate_cycle_sequences
from .split import split_fixed_test, kfold_trainval
from .net import CycleModel, CycleRNN, MensesHead, OvulationHeadFwd, OvulationHeadFull
from .losses import masked_menses_loss, masked_ovulation_bce, total_loss
from .train import run_stage1, run_stage2, train_epoch, eval_epoch
from .eval_metrics import evaluate, menses_mae, menses_accuracy_within_k, ovulation_correlation
