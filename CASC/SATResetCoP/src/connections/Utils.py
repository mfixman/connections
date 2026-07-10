from enum import StrEnum, auto

class UncaseEnum(StrEnum):
    @classmethod
    def _missing_(cls, value):
        try:
            return next(x for x in cls if x.name.lower() == value.lower())
        except StopIteration:
            raise LookupError(f'"{value}" not a valid {cls.__name__}')

    def __str__(self):
        return self.name
