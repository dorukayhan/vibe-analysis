print("importing pandas...")
import pandas as pd
import sys
from pathlib import Path
from typing import Final

type Confusion = dict[tuple[bool, bool], pd.Series[bool]]
type ConfusionCounts = dict[tuple[bool, bool], int]

TRIAL_ADDED_RELEASE_UNGRASP: Final[int] = 135 # told llm "release tissue" means UngraspTissue from this trial onward
TRIALS_CINCH_CHANGE: Final[frozenset[int]] = frozenset({146, 147}) # told llm "cinch" may be misheard as "change" on these trials
HEADER_WIDTH: Final[int] = 80

def main(fcalls: Path) -> None:
    out: Path = Path("assistant_testing") # should be gitignored
    out.mkdir(exist_ok=True)
    calls: pd.DataFrame = pd.read_csv(fcalls).dropna(subset=["Intent"]) # two [BLANK_AUDIO] rows with no intent
    # validate the logic in the spreadsheet on gdrive
    if (calls["CorrectInterpretation"] != (calls["Intent"] == calls["Interpretation"])).any():
        print("mismatch between CorrectInterpretation and actual interpretations!", file=sys.stderr)
        return
    # trials where llm's prompt changed
    release_means_ungrasp: pd.Series[bool] = calls["Trial"] >= TRIAL_ADDED_RELEASE_UNGRASP
    change_means_cinch: pd.Series[bool] = calls["Trial"].isin(TRIALS_CINCH_CHANGE)
    # ok now run the battery
    print_h1("ALL TRIALS")
    analyze(calls, out / "all trials")
    print_h1("BEFORE CLARIFYING RELEASE MEANS UNGRASP")
    analyze(calls.loc[~release_means_ungrasp], out / "before clarifying release means ungrasp")
    print_h1("AFTER CLARIFYING RELEASE MEANS UNGRASP")
    analyze(calls.loc[release_means_ungrasp], out / "after clarifying release means ungrasp")
    print_h1("EXCLUDING WHEN I BROKE LLM WITH \"CHANGE MEANS CINCH\"")
    analyze(calls.loc[~change_means_cinch], out / "except change-cinch fumble")
    print_h1("BEFORE RELEASE=UNGRASP, EXCEPT CHANGE=CINCH")
    analyze(calls.loc[~release_means_ungrasp & ~change_means_cinch], out / "before release-ungrasp, except change-cinch")
    print_h1("AFTER RELEASE=UNGRASP, EXCEPT CHANGE=CINCH")
    analyze(calls.loc[release_means_ungrasp & ~change_means_cinch], out / "after release-ungrasp, except change-cinch")

def analyze(calls: pd.DataFrame, out: Path) -> None:
    out.mkdir(exist_ok=True)
    # count all the correct- and mis-understandings, save misunderstandings
    confusion_all: Confusion = get_ai_confusion(calls)
    calls_per_success: dict[tuple[bool, bool], pd.DataFrame] = {k: calls.loc[v] for k, v in confusion_all.items()}
    print_ai_confusion(confusion_all, "ALL")
    calls_per_success[(True, False)].to_csv(out / "stt pass llm fail.csv", index=False)
    calls_per_success[(False, True)].to_csv(out / "stt fail llm pass.csv", index=False)
    calls_per_success[(False, False)].to_csv(out / "stt fail llm fail.csv", index=False)
    # count misunderstandings in each trial
    confusion_per_trial: dict[int, Confusion] = {trial: get_ai_confusion(calls.loc[calls["Trial"] == trial]) for trial in calls["Trial"].unique()}
    for t in confusion_per_trial:
        print_ai_confusion(confusion_per_trial[t], f"TRIAL {t}")
    pd.DataFrame({k: to_counts(v) for k, v in confusion_per_trial.items()}).to_csv(out / "per-trial confusions.csv")
    # which commands did the llm misunderstood?
    print_h2("LLM MISUNDERSTANDINGS")
    print("commands that whisper understood but llm didn't:")
    print(calls_per_success[(True, False)].value_counts(subset=["Intent", "Interpretation"]).sort_index())
    print("commands that neither whisper nor llm understood:")
    print(calls_per_success[(False, False)].value_counts(subset=["Intent", "Interpretation"]).sort_index())

def print_h2(header: str) -> None:
    print(header.center(HEADER_WIDTH, "*"))

def print_h1(header: str) -> None:
    print(header.center(2*HEADER_WIDTH, "*"))

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
    print_h2(header)
    length = sum(counts.values())
    print(f"{length} calls total")
    print(f"correct understanding in {counts[(True, True)]} ({counts[(True, True)]*100 / length}%) calls")
    print(f"llm failed in {counts[(True, False)]} ({counts[(True, False)]*100 / length}%) calls")
    print(f"somehow only whisper failed in {counts[(False, True)]} ({counts[(False, True)]*100 / length}%) calls")
    print(f"both ais failed in {counts[(False, False)]} ({counts[(False, False)]*100 / length}%) calls")

if __name__ == "__main__":
    main(Path(sys.argv[1]))