import pytest
import sp_main


def test_study_integrity():
    study = sp_main.main()
    assert study.best_value is not None
    max_trial = study.trials[-1]
    min_trial = study.trials[0]
    for trial in range(min_trial.number, max_trial.number + 1):
        assert study.trials[trial].user_attrs.get("depth") < 13
        if study.trials[trial].user_attrs.get("unused_features") is not None:
            unused_features = study.trials[trial].user_attrs.get("unused_features", [[]])
            total_num_unused_features = sum(list(map(len, unused_features)))
            assert len(unused_features) == 0
