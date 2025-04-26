from verifiers.rubrics import Rubric
from typing import List


class MathVerifyRubric(Rubric):
    def __init__(self):
        from verifiers.rubrics import is_latex_equal

        def latex_equal_reward_func(completions, answer, **kwargs) -> List[float | None]:
            return [1.0 if is_latex_equal(r, a) else 0.0 for r, a in zip(completions, answer)]

        self.reward_funcs = [
            latex_equal_reward_func
        ]
        self.reward_weights = [
            1
        ]
