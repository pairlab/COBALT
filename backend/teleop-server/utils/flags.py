"""
This file is a dirty way to handle any enums that we need.
"""


class ResetFlag:
    NO_RESET = 0
    USER_RESET = 1

    @staticmethod
    def description(code):
        descriptions = ["ResetFlag: NO_RESET", "ResetFlag: USER_RESET"]
        return descriptions[code]
