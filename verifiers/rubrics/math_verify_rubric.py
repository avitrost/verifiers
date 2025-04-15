from verifiers.rubrics import Rubric


class MathVerifyRubric(Rubric):
    def __init__(self):
        from verifiers.rubrics import is_latex_equal
        self.reward_funcs = [
            is_latex_equal
        ]
        self.reward_weights = [
            1
        ]
