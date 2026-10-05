from enum import StrEnum

class Choice(StrEnum):
    def wire_value(self) -> str:
        return self.value.lower()

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for item in cls:
                if value.casefold() in (item.value.casefold(), item.wire_value()):
                    return item

        return None

class ProverPolicy(Choice):
    SatResetCoP = "SatResetCoP"
    SatCoP = "SatCoP"

class GraphInputKind(Choice):
    Full = "Full"
    NoComplements = "NoComplements"
    NoTerms = "NoTerms"

    def wire_value(self) -> str:
        return self.value.replace("No", "no-").lower()

class GuidanceMode(Choice):
    Weighted = "Weighted"
    Strict = "Strict"
    Base = "Base"

def plain_values(value):
    if isinstance(value, Choice):
        return value.wire_value()

    if isinstance(value, dict):
        return {plain_values(key): plain_values(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return type(value)(plain_values(item) for item in value)

    return value
