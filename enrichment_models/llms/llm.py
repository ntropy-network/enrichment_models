class LLM:
    def predict(
        self, instructions: str | list[str], inputs: str | list[str]
    ) -> str | list[str]:
        raise NotImplementedError()
