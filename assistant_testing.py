print("importing pandas...")
import pandas as pd
import sys
from pathlib import Path

type Confusion = dict[tuple[bool, bool], pd.Series[bool]]
type ConfusionCounts = dict[tuple[bool, bool], int]

def main(calls: Path) -> None:
    out: Path = Path("assistant_testing") # should be gitignored
    out.mkdir(exist_ok=True)
    df: pd.DataFrame = pd.read_csv(calls).dropna(subset=["Intent"]) # two [BLANK_AUDIO] rows with no intent
    # validate the logic in the spreadsheet on gdrive
    if (df["CorrectInterpretation"] != (df["Intent"] == df["Interpretation"])).any():
        print("mismatch between CorrectInterpretation and actual interpretations!", file=sys.stderr)
        return
    confusion_all = get_ai_confusion(df)
    # for k in confusion:
    #     print(k)
    #     print(df.loc[confusion[k]])
    print_ai_confusion(confusion_all, "ALL")
    confusion_per_trial: dict[int, Confusion] = {trial: get_ai_confusion(df.loc[df["Trial"] == trial]) for trial in df["Trial"].unique()}
    for t in confusion_per_trial:
        print_ai_confusion(confusion_per_trial[t], f"TRIAL {t}")
    print(pd.DataFrame({k: to_counts(v) for k, v in confusion_per_trial.items()}))

def get_ai_confusion(df: pd.DataFrame) -> Confusion:
    stt_pass: pd.Series[bool] = df["CorrectTranscription"]
    llm_pass: pd.Series[bool] = df["Intent"] == df["Interpretation"]
    # can also infer llm_pass being a Series[bool] but spell that out for consistency
    return {
        (True, True): stt_pass & llm_pass,
        (True, False): stt_pass & ~llm_pass,
        (False, True): ~stt_pass & llm_pass,
        (False, False): ~stt_pass & ~llm_pass
    }

def to_counts(confusion: Confusion) -> ConfusionCounts:
    return {k: v.value_counts().get(True, 0) for k, v in confusion.items()}

def print_ai_confusion(confusion: Confusion, header: str) -> None:
    counts = to_counts(confusion)
    print(header.center(80, "*"))
    length = sum(counts.values())
    print(f"{length} calls total")
    print(f"correct understanding in {counts[(True, True)]} ({counts[(True, True)]*100 / length}%) calls")
    print(f"llm failed in {counts[(True, False)]} ({counts[(True, False)]*100 / length}%) calls")
    print(f"somehow only whisper failed in {counts[(False, True)]} ({counts[(False, True)]*100 / length}%) calls")
    print(f"both ais failed in {counts[(False, False)]} ({counts[(False, False)]*100 / length}%) calls")

if __name__ == "__main__":
    main(Path(sys.argv[1]))