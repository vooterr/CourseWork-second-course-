def normalize_prediction(output: str, classes: list[str]) -> str:
    output = output.lower().strip()

    for cls in classes:
        if cls.lower() in output:
            return cls

    return "unknown"
