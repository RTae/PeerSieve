from __future__ import annotations

from experiments.phase1 import figroles as fr


def test_figroles_attack_and_ablation_mappings_are_complete() -> None:
    expected_attacks = set(fr.ATTACK_ORDER)
    assert expected_attacks == {"none", "label_flip_all", "model_poison", "random_noise"}
    assert set(fr.ATTACK_COLOR) == expected_attacks
    assert set(fr.ATTACK_MARKER) == expected_attacks
    assert set(fr.ATTACK_LINESTYLE) == expected_attacks
    assert set(fr.ATTACK_LABEL) == expected_attacks
    assert set(fr.ABLATION_COLOR) == {"tg_stab", "tg_a03", "tg_a07", "tg_euclidean", "tg_no_relu"}
