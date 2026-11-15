# Unified answer verifier v2
New probes and outcome-only training runs explicitly select
rlvr.verifier_version=rlvr-answer-v2. The shared CPU implementation is
rlvr.answer_verifier.verify_answer. The reward adapter delegates to that
implementation. Legacy defaults remain available for frozen-contract replay;
dataset and trainer verifier versions must agree.

Final-answer extraction never reads gold. Contradictory conclusions are
ambiguous. Numeric comparison uses bounded Decimal/Fraction parsing, LaTeX
fractions, scientific notation and Unicode digits. Percent semantics must be
declared. No unit conversion or numeric tolerance is inferred.

Correctness, format and native termination are independent. DPO pairs require
complete, unambiguous outputs. A truncated but explicitly correct final can
score one, while remaining ineligible for DPO. V2 refuses unresolved execution
or backend failure as a training reward. No extra format/language/length
reward is introduced.

CPU validation:
python -m pytest tests/test_answer_verifier_v2.py tests/test_contracts_rewards.py -q
CPU result: 96 passed. The real-output pilot also covers Tibetan shad-delimited
answer headings and terminal option labels; explanations are not scanned for a
gold-matching label. Probe and reward adapters use the same implementation.
GPU generation and teacher-forcing evidence is stored with the finetuning probe.
