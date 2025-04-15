from verifiers.rubrics import Rubric, is_latex_equal


class MathVerifyRubric(Rubric):
    def __init__(self):
        self.reward_funcs = [
            is_latex_equal
        ]
