print("importing pandas...")
import pandas as pd
import sys
from pathlib import Path
from typing import Iterable

def main(calls: Path) -> None:
    df: pd.DataFrame = pd.read_csv(calls).dropna(subset=["Intent"]) # two [BLANK_AUDIO] rows with no intent
    # validate the logic in the spreadsheet on gdrive
    if (df["CorrectInterpretation"] != (df["Intent"] == df["Interpretation"])).any():
        print("mismatch between CorrectInterpretation and actual interpretations!", file=sys.stderr)
        return
    stt_pass: pd.Series[bool] = df["CorrectTranscription"]
    llm_pass: pd.Series[bool] = df["Intent"] == df["Interpretation"]
    # can also infer llm_pass being a Series[bool] but spell that out for consistency
    stt_llm_confusion: dict[tuple[bool, bool], pd.Series[bool]] = {
        (True, True): stt_pass & llm_pass,
        (True, False): stt_pass & ~llm_pass,
        (False, True): ~stt_pass & llm_pass,
        (False, False): ~stt_pass & ~llm_pass
    }
    print({k: v.value_counts()[True] for k, v in stt_llm_confusion.items()})
    print({k: v.value_counts()[True] / len(df) for k, v in stt_llm_confusion.items()})
    for k in stt_llm_confusion:
        print(k)
        print(df.loc[stt_llm_confusion[k]])

if __name__ == "__main__":
    main(Path(sys.argv[1]))