import sys
print("importing pandas...", file=sys.stderr)
import pandas as pd
import scipy
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
    analyze(calls, out / "all trials", all_trials=True)
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
    print_h1("PERFORMANCE (ALL TIMES IN SECONDS)")
    analyze_perf(calls, out / "performance")

def analyze(calls: pd.DataFrame, out: Path, *, all_trials: bool=False) -> None:
    out.mkdir(exist_ok=True)
    # count all the correct- and mis-understandings, save misunderstandings
    confusion_all: Confusion = get_ai_confusion(calls)
    calls_per_success: dict[tuple[bool, bool], pd.DataFrame] = {k: calls.loc[v] for k, v in confusion_all.items()}
    assert sum(map(len, calls_per_success.values())) == len(calls)
    print_ai_confusion(confusion_all, "ALL")
    calls_per_success[(True, False)].to_csv(out / "stt pass llm fail.csv", index=False)
    calls_per_success[(False, True)].to_csv(out / "stt fail llm pass.csv", index=False)
    calls_per_success[(False, False)].to_csv(out / "stt fail llm fail.csv", index=False)
    # count misunderstandings in each trial, or just list which trials were considered
    if all_trials:
        confusion_per_trial: dict[int, Confusion] = {trial: get_ai_confusion(calls.loc[calls["Trial"] == trial]) for trial in calls["Trial"].unique()}
        for t in confusion_per_trial:
            print_ai_confusion(confusion_per_trial[t], f"TRIAL {t}")
        pd.DataFrame({k: to_counts(v) for k, v in confusion_per_trial.items()}).to_csv(out / "per-trial confusions.csv")
    else:
        print(f"looked into trials {calls["Trial"].unique()}")
    # which commands did the llm misunderstood?
    print_h2("MISUNDERSTANDINGS")
    print("fraction of commands that whisper understood but llm didn't:")
    print(calls_per_success[(True, False)].value_counts(subset=["Intent", "Interpretation"]).sort_index() / len(calls))
    print("fraction of commands that whisper didn't understand but llm somehow did:")
    print(calls_per_success[(False, True)].value_counts(subset=["Intent"]) / len(calls))
    print("fraction of commands that neither whisper nor llm understood:")
    print(calls_per_success[(False, False)].value_counts(subset=["Intent", "Interpretation"]).sort_index() / len(calls))
    if all_trials:
        print("number of commands that whisper misheard:")
        whisper_failed: pd.DataFrame = calls.loc[~calls["CorrectTranscription"]]
        print(whisper_failed.value_counts(subset=["ActualCommand"]))

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

def analyze_perf(calls: pd.DataFrame, out: Path) -> None:
    out.mkdir(exist_ok=True)
    domain_of_discourse: list[pd.Series[int | float]] = [calls["Trial"], calls["LLMStartTime"] - calls["STTStartTime"], calls["LLMEndTime"] - calls["LLMStartTime"]]
    perf: pd.DataFrame = pd.concat(domain_of_discourse, axis="columns").rename(columns={0: "STTInferenceTime", 1: "LLMInferenceTime"})
    perf.to_csv(out / "raw.csv", index=False)
    # first call of each trial, where for some reason llm is much slower
    first_calls: pd.DataFrame = perf.drop_duplicates(subset=["Trial"], keep="first", inplace=False)
    # subsequent calls
    nonfirst_calls: pd.DataFrame = perf.drop(first_calls.index, axis="index")
    print_h2("ALL")
    print("summary stats of all calls' whisper (stt) and llm inference times:")
    print(perf[["STTInferenceTime", "LLMInferenceTime"]].describe())
    print("summary stats of each trial's first call:")
    print(first_calls[["STTInferenceTime", "LLMInferenceTime"]].describe())
    print("summary stats of each trial's subsequent calls:")
    print(nonfirst_calls[["STTInferenceTime", "LLMInferenceTime"]].describe())
    print("t-testing first and subsequent calls' llm inference times...", end="", flush=True)
    llm_t_test = scipy.stats.ttest_ind(first_calls["LLMInferenceTime"], nonfirst_calls["LLMInferenceTime"], equal_var=False, nan_policy="raise", alternative="greater")
    print(f"done!\n{llm_t_test}")
    print("t-testing first and subsequent calls' whisper inference times just because...", end="", flush=True)
    stt_t_test = scipy.stats.ttest_ind(first_calls["STTInferenceTime"], nonfirst_calls["STTInferenceTime"], equal_var=False, nan_policy="raise")
    print(f"done!\n{stt_t_test}")
    # also save summary stats. finding t-tests in stdout easy enough by searching for "TtestResult"
    perf[["STTInferenceTime", "LLMInferenceTime"]].describe().to_csv(out / "summary overall.csv")
    first_calls[["STTInferenceTime", "LLMInferenceTime"]].describe().to_csv(out / "summary first.csv")
    nonfirst_calls[["STTInferenceTime", "LLMInferenceTime"]].describe().to_csv(out / "summary nonfirst.csv")
    # now consider each trial
    for trial in calls["Trial"].unique():
        print_h2(f"TRIAL {trial}")
        first: pd.Series[float] = first_calls.loc[first_calls["Trial"] == trial].iloc[0]
        print("first call:")
        print(first)
        nonfirst: pd.DataFrame = nonfirst_calls.loc[nonfirst_calls["Trial"] == trial]
        if nonfirst.empty:
            print("no other calls in this trial!")
        else:
            stats = nonfirst[["STTInferenceTime", "LLMInferenceTime"]].describe(percentiles=[0.25, 0.5, 0.75, 0.99])
            stats.to_csv(out / f"summary nonfirst trial {trial}.csv")
            print("in subsequent calls:")
            print(f"whisper inference averaged {stats["STTInferenceTime"]["mean"]} s (sigma={stats['STTInferenceTime']['std']} s), first call is {(first["STTInferenceTime"]-stats["STTInferenceTime"]["mean"])/stats['STTInferenceTime']['std']} sigma away")
            print(f"llm inference averaged {stats["LLMInferenceTime"]["mean"]} s (sigma={stats['LLMInferenceTime']['std']} s), first call is {(first["LLMInferenceTime"]-stats["LLMInferenceTime"]["mean"])/stats['LLMInferenceTime']['std']} sigma away")

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