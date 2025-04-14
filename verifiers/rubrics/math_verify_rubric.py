from verifiers.rubrics import Rubric
from verifiers.utils.math_utils import is_latex_equal


class MathVerifyRubric(Rubric):
    def __init__(self):
        self.reward_funcs = [
            is_latex_equal
        ]
