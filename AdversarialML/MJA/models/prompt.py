from dataclasses import dataclass
from typing import Optional, Sequence

from common.loadable_yaml import LoadableYAML  # your helper

@dataclass
class Prompt(LoadableYAML):
    # Template text is required
    prompt: str

    # Names of placeholders that *may* need values later
    parameters: Optional[Sequence[str]] = None

    def render(self, **values: str) -> str:
        """
        Format the template with the supplied parameter values.
        Raises if you pass unexpected or missing keys (helps during dev).
        """
        self._validate(values)
        return self.prompt.format(**values)

    def with_parameters(self, **values: str) -> "Prompt":
        """
        Return a *new* Prompt that carries the filled-in dict
        (handy if you chain operations).
        """
        self._validate(values)
        return Prompt(prompt=self.prompt, parameters=values.keys())._replace(values)

    # ----------------------------------------------------------------------

    # ----- internal utilities ---------------------------------------------
    def _validate(self, values: dict) -> None:
        allowed = set(self.parameters or [])
        if allowed and (missing := allowed - values.keys()):
            raise ValueError(f"Missing parameters: {', '.join(sorted(missing))}")
        if allowed and (extra := values.keys() - allowed):
            raise ValueError(f"Unexpected parameters: {', '.join(sorted(extra))}")

    def _replace(self, values: dict) -> "Prompt":
        # convenience wrapper so you can keep the parameters dict if you like
        object.__setattr__(self, "parameters", values)  # dataclass hack
        return self
