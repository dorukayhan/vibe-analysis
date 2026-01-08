import doga_movement_calc as dmc
import itertools
import math
import numpy as np
import pandas as pd
import scipy
import statistics
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from matplotlib import pyplot as plt
from scipy.stats._result_classes import TtestResult
from typing import Optional

type ATBS = dict[int, float]

# columns of survey results
auto_headers: list[str] = ["Start Date, End Date, Recorded Date, Response ID"]
pre_survey: list[str] = ["Age", "Gender", "Hand Dominance", "Ethnicity", "Years as Fellow", "Years as Advanced Endoscopy Fellow", "Years in Practice", "Endoscopies in Last 6 Months", "Total Endoscopies", "Total Endoscopic Suturing Cases", "Total ESGs", "Is Gamer", "Weekly Time Spent Gaming", "Used VR Before", "Weekly Time Spent Gaming in VR", "Used VR Surgical/Endoscopy Simulator Before", "VR Simulator Previously Used"]
post_survey: list[str] = ["Marking Look Realism", "Grasping Look Realism", "Suturing Needle Look Realism", "Suturing Overall Look Realism", "Suturing Overall Feel Realism", "Anchor Exchange Look Realism", "Cinch Look Realism", "Instrument Handling Look Realism", "Anatomy Look Realism", "Overall Look Realism", "Overall Mechanical Realism", "Bleeding Realism", "Haptic Realism", "Haptic Usefulness", "Tube Responsiveness/Accuracy", "Suture Trigger Responsiveness", "Helix Responsiveness", "Overall Feel Realism", "Overall Immersiveness", "VR Effectiveness", "Bird's Eye Effectiveness", "Overall Responsiveness", "Hand-Eye Coordination Usefulness", "Overall Training Usefulness", "Assessment Trustworthiness", "OR Assistant Importance", "Virtual Assistant Helix Importance", "Virtual Assistant Cinch Importance", "Virtual Assistant Overall Importance"]
other: list[str] = ["Realism Suggestions/Other"]

# type checked file names!
class CSVData(StrEnum):
    ArgonMarkInfo = "ArgonMarkInfo"
    ArgonParallelInfo = "ArgonParallelInfo"
    BenchmarkInfo = "BenchmarkInfo"
    BleedingInfo = "BleedingInfo"
    FrameRates = "FrameRates"
    GazeInfo = "GazeInfo"
    HMDPosition = "HMDPosition"
    SutureInfo = "SutureInfo"
    TechniqueInfo = "TechniqueInfo"
nodata: dict[CSVData, pd.DataFrame] = {
    CSVData.ArgonMarkInfo: pd.DataFrame(columns=["Zone","ArgonTime","DistanceToZone","MarkPosition(x)","MarkPosition(y)","MarkPosition(z)"]),
    CSVData.ArgonParallelInfo: pd.DataFrame(columns=["CrossProduct(x)","CrossProduct(y)","CrossProduct(z)"]),
    CSVData.BenchmarkInfo: pd.DataFrame(columns=["SutureSetCount","SutureCount","FPS","RenderTime","SolverTime"]),
    CSVData.BleedingInfo: pd.DataFrame(columns=["BleedingType","BleedingIndex","BleedingStartTime","BleedingFinishTime","BleedingPosition(x)","BleedingPosition(y)","BleedingPosition(z)"]),
    CSVData.FrameRates: pd.DataFrame(columns=["Seconds","Frame"]),
    CSVData.GazeInfo: pd.DataFrame(columns=["Seconds","Target","Status"]),
    CSVData.HMDPosition: pd.DataFrame(columns=["Seconds","PositionX","PositionY","PositionZ","RotationX","RotationY","RotationZ","QuatX","QuatY","QuatZ","QuatW"]),
    CSVData.SutureInfo: pd.DataFrame(columns=["SutureSetCount","SutureCount","SutureIndex","SutureTime","IsAnchorExchangeDone","StomachPart","SuturePosition(x)","SuturePosition(y)","SuturePosition(z)"]),
    CSVData.TechniqueInfo: pd.DataFrame(columns=["PatternType","SutureSetCount","ArrowPlacingTIme"])
}

# participants and trials for each. each index into the lists matches the participant's index in the survey df
# DO NOT REORDER THE SURVEY CSV IT MUST BE IN CHRONOLOGICAL ORDER
participants: list[str] = ["W-1", "W-2", "W-3", "W-4", "W-5", "Th-1", "Th-2", "mark", "Th-3", "F-1", "F-2", "F-3", "F-4", "F-5"]
trials: list[list[int]] = [
    [1], # W-1
    [2], # W-2
    [3], # W-3
    [4, 5], # W-4
    [6, 7], # W-5
    [8], # Th-1
    [11], # Th-2
    [13], # mark
    [14, 15, 16, 17], # Th-3
    [19], # F-1
    [20, 22], # F-2
    [23, 24], # F-3
    [25, 26, 27], # F-4
    [29, 30, 31] # F-5
]

# valid gaze targets that collate_gaze_info will consider
gaze_targets: list[str] = ["tv", "tv_stomachpos", "Instructions_TV", "floor"]

# correct stomach zone order for suturing in perf_scoring
zone_order: list[str] = ["Anterior", "GreaterCurvature", "Posterior"]

def read_csvdata(trial: int, data: CSVData) -> pd.DataFrame:
    """read the given trial's given csv recording"""
    try:
        return pd.read_csv(f"vibe study csvdata/Trial {trial}/{data}.csv")
    except pd.errors.EmptyDataError:
        return nodata[data] # return dummy empty df if a simulator crash inexplicably truncated the csv to oblivion

def avg_time_between_sutures(participant: int) -> float:
    """figure out how much time the given participant takes between two sutures on average"""
    # several lists instead of generators or one giant list comprehension to allow better debugging
    suture_times: list[pd.Series] = [read_csvdata(trial, CSVData.SutureInfo)["SutureTime"] for trial in trials[participant]]
    times_between_sutures: list[pd.Series] = [trial.diff() for trial in suture_times if not trial.diff().empty]
    tbspd: pd.Series = pd.concat(times_between_sutures).dropna() if times_between_sutures else pd.Series()
    return float(tbspd.mean()) if not tbspd.empty else math.nan

def avg_hmd_velocity_etc(participant: int) -> tuple[float, float, float]:
    """figure out the given participant's average hmd velocity, acceleration, and jerk"""
    derivs: list[tuple[list[dmc.Vec3], list[dmc.Vec3], list[dmc.Vec3]]] = [
        dmc.calculate_position_derivatives(f"vibe study csvdata/Trial {trial}/{CSVData.HMDPosition}.csv")
        for trial in trials[participant]
    ]
    avg = statistics.mean
    hyp = math.hypot
    velocities = itertools.chain.from_iterable((v for v, a, j in derivs))
    accelerations = itertools.chain.from_iterable((a for v, a, j in derivs))
    jerks = itertools.chain.from_iterable((j for v, a, j in derivs))
    return avg((hyp(*v) for v in velocities)), avg((hyp(*a) for a in accelerations)), avg((hyp(*j) for j in jerks))

def hmd_t_tests(group_1: pd.DataFrame, group_2: pd.DataFrame, testname: Optional[str]=None) -> list[tuple[str, TtestResult]]:
    """run t-tests on hmd movement data"""
    velocity_test = WELCH(group_1["velocity"], group_2["velocity"])
    acceleration_test = WELCH(group_1["acceleration"], group_2["acceleration"])
    jerk_test = WELCH(group_1["jerk"], group_2["jerk"])
    if testname:
        print_t_test(f"{testname} hmd velocity", velocity_test)
        print_t_test(f"{testname} hmd acceleration", acceleration_test)
        print_t_test(f"{testname} hmd jerk", jerk_test)
    return [velocity_test, acceleration_test, jerk_test]

def gaze_t_tests(group_1: pd.DataFrame, group_2: pd.DataFrame, testname: Optional[str]=None) -> list[tuple[str, TtestResult]]:
    """run t-tests on gaze info"""
    endo_cam = WELCH(group_1["tv"], group_2["tv"])
    wide_stomach_cam = WELCH(group_1["tv_stomachpos"], group_2["tv_stomachpos"])
    instructions = WELCH(group_1["Instructions_TV"], group_2["Instructions_TV"])
    if testname:
        print_t_test(f"{testname} gaze endo", endo_cam)
        print_t_test(f"{testname} gaze stomach", wide_stomach_cam)
        print_t_test(f"{testname} gaze instructions", instructions)
    return [endo_cam, wide_stomach_cam, instructions]

def save_boxplot(df: pd.DataFrame, filename: str, **kwargs):
    """boxplot the given df without mucking with matplotlib. kwargs go to df.boxplot"""
    print(df)
    fig, ax = plt.subplots(layout="constrained")
    df.boxplot(ax=ax, **kwargs)
    plt.savefig(filename)
    plt.close()

def save_survey_summary(survey: pd.DataFrame, filename_no_ext: str, **more_boxplot_kwargs):
    """summarize and boxplot the given post-survey results. kwargs go to save_boxplot"""
    survey.describe().transpose().to_csv(filename_no_ext + ".csv")
    save_boxplot(survey, filename_no_ext + ".svg", vert=False, **more_boxplot_kwargs)

def collate_gaze_info(participant: int) -> dict[str, float]:
    """figure out how much time the given participant's hmd gaze spent on different things, normalized over all trials"""
    ret: Counter = Counter()
    for trial in trials[participant]:
        looking_at: str | None = None
        last_enter: float | None = None
        last_click: float | None = None
        for row in read_csvdata(trial, CSVData.GazeInfo).itertuples():
            # have row.Seconds, row.Target, row.Status. stop reading if something goes wrong
            try:
                if row.Status == "Enter" and not looking_at:
                    last_enter = row.Seconds
                    last_click = row.Seconds
                    looking_at = row.Target
                elif "Click".startswith(row.Status) and row.Target == looking_at:
                    last_click = row.Seconds
                elif row.Status == "Exit" and row.Target == looking_at:
                    ret[looking_at] += row.Seconds - last_enter
                    looking_at = None
                    last_enter = None
                    last_click = None
                else:
                    # raise ValueError(f"something is off with trial {trial}'s gazeinfo, got {row} while looking at {looking_at}")
                    break
            except TypeError: # from "Click".startswith
                break
        # catch simulation ending while user was looking at something
        if looking_at:
            ret[looking_at] += last_click - last_enter
    total: float = sum([ret[target] for target in gaze_targets])
    return {target: ret[target] / total for target in gaze_targets} # filter out other stuff that wasn't supposed to be gaze interacted

def video_scoring(row: pd.Series, marking_desc: pd.Series, suture_desc: pd.Series) -> int:
    return sum(individual_video_scoring(row, marking_desc, suture_desc).values())

def individual_video_scoring(row: pd.Series, marking_desc: pd.Series, suture_desc: pd.Series) -> dict[str, int]:
    """calculate video metric from one row in video metrics.csv"""
    ret = {}
    # retried?
    ret["Trials Minus Crashes"] = {
        # switch statement at home
        1: 0,
        2: 2
    }[row["Trials Minus Crashes"]]
    # progress discounting crashes?
    fr = row["Failure Reason"]
    ret["Procedure Progress"] = 0
    if fr == "gave up" or fr == "left stomach":
        ret["Procedure Progress"] = {
            "complete": 0,
            "cinched": 5,
            "partial suture": 15,
            "full marking": 25
        }[row["Procedure Progress"]]
    # left stomach during marking?
    ret["Marking: Left Stomach"] = 0
    if row["Marking: Left Stomach"] == "yes":
        ret["Marking: Left Stomach"] = {
            "yes": 2,
            "no": 5
        }[row["Marking: Returned After Leaving"]]
    # marking type?
    # score += {
    #     "parallel lines": 0,
    #     "nonparallel lines": 2,
    #     "dots": 5,
    #     "something else": 10
    # }[row["Marking: Mark Type"]]
    # marking time taken
    mtime = row["Marking: Time Taken"]
    ret["Marking: Time Taken"] = 0
    if mtime <= marking_desc["25%"]:
        ret["Marking: Time Taken"] = 0
    elif mtime <= marking_desc["50%"]:
        ret["Marking: Time Taken"] = 1
    elif mtime <= marking_desc["75%"]:
        ret["Marking: Time Taken"] = 2
    else:
        ret["Marking: Time Taken"] = 3
    # left stomach during suturing?
    ret["Suture: Left Stomach"] = 0
    if row["Suture: Left Stomach"] == "yes":
        ret["Suture: Left Stomach"] = {
            "yes": 2,
            "no": 5
        }[row["Suture: Returned After Leaving"]]
    # proper u suture?
    ret["Suture: U-Shaped Suture"] = {
        "yes": 0,
        "no": 5
    }[row["Suture: U-Shaped Suture"]]
    # enough bites?
    bites = row["Suture: Successful Bites"]
    ret["Suture: Successful Bites"] = 6 - bites
    # remembered to put the suture on the needle before first bite?
    ret["Suture: Anchor Exchange Before First Bite"] = {
        "yes": 0,
        "no": 2
    }[row["Suture: Anchor Exchange Before First Bite"]]
    # followed own marking when biting?
    bnm = row["Suture: Bites Near Marks"]
    ret["Suture: Bites Near Marks"] = 0
    if not math.isnan(bnm):
        ret["Suture: Bites Near Marks"] = bites - bnm
    # squeezed too hard when biting and accidentally anchor exchanged?
    ret["Suture: Accidental Anchor Exchanges"] = row["Suture: Accidental Anchor Exchanges"]
    # cinched properly?
    ret["Suture: Full Cinch"] = 0
    if row["Procedure Progress"] == "complete":
        ret["Suture: Full Cinch"] = {
            "yes": 0,
            "no": 5
        }[row["Suture: Full Cinch"]]
    # dropped cinch after suturing?
    ret["Suture: Dropped Cinch"] = {
        "near ttag": 0,
        "not near ttag": 3,
        "no": 0
    }[row["Suture: Dropped Cinch"]]
    # suture time taken
    stime = row["Suture: Time Taken"]
    ret["Suture: Time Taken"] = 0
    if stime <= suture_desc["25%"]:
        ret["Suture: Time Taken"] = 0
    elif stime <= suture_desc["50%"]:
        ret["Suture: Time Taken"] = 3
    elif stime <= suture_desc["75%"]:
        ret["Suture: Time Taken"] = 6
    else:
        ret["Suture: Time Taken"] = 9
    
    return ret

def perf_scoring(participant: int) -> int:
    """(try to?) calculate performance metric score from csvdata.
    same metrics as in preliminary validation"""
    csvdata: dict[CSVData, list[pd.DataFrame]] = {
        data: [read_csvdata(trial, data) for trial in trials[participant]]
        for data in CSVData
    }
    score = 0
    # marking
    # search for markings in given zones across all trials
    if not any((ami["Zone"].str.contains("Anterior", regex=False).any() for ami in csvdata[CSVData.ArgonMarkInfo])):
        score += 5 
    if not any((ami["Zone"].str.contains("Posterior", regex=False).any() for ami in csvdata[CSVData.ArgonMarkInfo])):
        score += 5
    if not any((ami["Zone"].str.contains("GreaterCurvature", regex=False).any() for ami in csvdata[CSVData.ArgonMarkInfo])):
        score += 5
    # suture
    # penalize each suture that isn't near marking and getting the anterior-great curvature-posterior order wrong. do bleeding too
    suture_score = 0
    for i in range(len(trials[participant])):
        my_suture_score = 0
        marks = csvdata[CSVData.ArgonMarkInfo][i]
        sutures = csvdata[CSVData.SutureInfo][i]
        wounds = csvdata[CSVData.BleedingInfo][i]
        zones: list[str] = []
        for sidx, suture in sutures.iterrows():
            # is suture near marking? ignore if marking was skipped due to sim choke
            if not marks.empty:
                xdiff = (marks["MarkPosition(x)"] - suture["SuturePosition(x)"]).abs()
                ydiff = (marks["MarkPosition(y)"] - suture["SuturePosition(y)"]).abs()
                zdiff = (marks["MarkPosition(z)"] - suture["SuturePosition(z)"]).abs()
                my_suture_score += 0 if ((xdiff <= 0.05) & (ydiff <= 0.05) & (zdiff <= 0.05)).any() else 5
            # track zone order
            zone = suture["StomachPart"]
            if zone in zone_order and zone not in zones:
                zones.append(zone)
        # was anterior-gc-posterior order followed?
        if zones != zone_order:
            my_suture_score += 5
        for z in zone_order:
            if z not in zones:
                my_suture_score += 5
        # count stopped bleeding as bleeding episode that lasts 2 s or less, penalize unstopped bleeding
        if not wounds.empty:
            my_suture_score += 5 if ((wounds["BleedingFinishTime"] - wounds["BleedingStartTime"]) > 2).any() else 0
        suture_score = max(suture_score, my_suture_score)
    score += suture_score
    return score

# def longest_trial(parti)

def WELCH(group_1, group_2, **kwargs) -> tuple[str, TtestResult]:
    # 1-tailed welch's t-test. pick correct tail for the caller
    alt = "less" if statistics.mean(group_1) <= statistics.mean(group_2) else "greater"
    return alt, scipy.stats.ttest_ind(group_1, group_2, equal_var=False, nan_policy="raise", alternative=alt, **kwargs)

def print_t_test(test: str, result: tuple[str, TtestResult]):
    print(f"{test} t test: {result}")
    if result[1].pvalue < 0.05:
        print("!!!P<0.05 RESULT WOOOOOOOO!!!")

def main(argv: list[str]=[]) -> pd.DataFrame:
    survey: pd.DataFrame = pd.read_csv("vibe study.csv")
    # expert-novice split: at least 5000 endoscopies and 25 suturing cases
    # experts: pd.DataFrame = survey[(survey["Total Endoscopies"] >= 5000) & (survey["Total Endoscopic Suturing Cases"] >= 20)]
    # experts: pd.DataFrame = survey[survey["Total Endoscopic Suturing Cases"] >= 20]
    # experts: pd.DataFrame = survey[(survey["Years in Practice"] > 5) & (survey["Total Endoscopies"] > 1500) & (survey["Total Endoscopic Suturing Cases"] > 10)]
    experts: pd.DataFrame = survey[survey["Years in Practice"] > 0]
    novices: pd.DataFrame = survey.drop(experts.index) # literally survey minus experts
    # another split idea: gamer-nongamer
    gamers: pd.DataFrame = survey[survey["Is Gamer"] == "Yes"]
    nongamers: pd.DataFrame = survey.drop(gamers.index)
    # post questionnaire answers
    save_survey_summary(survey[post_survey], "post survey/everyone")
    save_survey_summary(experts[post_survey], "post survey/experts")
    save_survey_summary(novices[post_survey], "post survey/novices")
    save_survey_summary(gamers[post_survey], "post survey/gamers")
    save_survey_summary(nongamers[post_survey], "post survey/nongamers")
    # compare suture times between experts and novices, use welch's t test
    def times(participants: pd.Index) -> ATBS:
        raw: ATBS = {i: avg_time_between_sutures(i) for i in participants}
        return {k: v for k, v in raw.items() if math.isfinite(v)} # dropna() by hand
    atbs_expert: ATBS = times(experts.index)
    atbs_novice: ATBS = times(novices.index)
    atbs_gamer: ATBS = times(gamers.index)
    atbs_nongamer: ATBS = times(nongamers.index)
    # negative t-statistic = first list has lower mean than second list
    atbs_expert_vs_novice: tuple[str, TtestResult] = WELCH(list(atbs_expert.values()), list(atbs_novice.values()))
    atbs_gamer_vs_not: tuple[str, TtestResult] = WELCH(list(atbs_gamer.values()), list(atbs_nongamer.values()))
    print_t_test(f"expert-novice ATBS", atbs_expert_vs_novice)
    print_t_test(f"gamer-nongamer ATBS", atbs_gamer_vs_not)
    # metric t-tests
    # video metrics
    video_metrics: pd.DataFrame = pd.read_csv("video metrics.csv")
    quartiles: tuple[pd.Series, pd.Series] = (video_metrics["Marking: Time Taken"].describe(), video_metrics["Suture: Time Taken"].describe())
    video_metric_totals: pd.Series = video_metrics.apply(video_scoring, axis="columns", result_type="reduce", args=quartiles)
    # print(video_metric_totals)
    vmt_expert: pd.Series = video_metric_totals.loc[experts.index]
    vmt_novice: pd.Series = video_metric_totals.loc[novices.index]
    vmt_gamer: pd.Series = video_metric_totals.loc[gamers.index]
    vmt_nongamer: pd.Series = video_metric_totals.loc[nongamers.index]
    vmt_expert_vs_novice: tuple[str, TtestResult] = WELCH(vmt_expert, vmt_novice)
    vmt_gamer_vs_not: tuple[str, TtestResult] = WELCH(vmt_gamer, vmt_nongamer)
    print_t_test(f"expert-novice video metric", vmt_expert_vs_novice)
    print_t_test(f"gamer-nongamer video metric", vmt_gamer_vs_not)
    # just the time taken to do things?
    for phase in ("Marking", "Suture"):
        phasett_expert = video_metrics[f"{phase}: Time Taken"].loc[experts.index]
        phasett_novice = video_metrics[f"{phase}: Time Taken"].loc[novices.index]
        phasett_gamer = video_metrics[f"{phase}: Time Taken"].loc[gamers.index]
        phasett_nongamer = video_metrics[f"{phase}: Time Taken"].loc[nongamers.index]
        phasett_expert_vs_novice = WELCH(phasett_expert, phasett_novice)
        phasett_gamer_vs_not = WELCH(phasett_gamer, phasett_nongamer)
        print_t_test(f"expert-novice {phase.lower()} time taken", phasett_expert_vs_novice)
        print_t_test(f"gamer-nongamer {phase.lower()} time taken", phasett_gamer_vs_not)
    # auto performance metrics
    # WHY IS NOTHING STATISTICALLY SIGNIFICANT WHAT THE FUCK IS HAPPENING
    perf_metrics: pd.Series = pd.Series([perf_scoring(p) for p in survey.index])
    perf_expert: pd.Series = perf_metrics.loc[experts.index]
    perf_novice: pd.Series = perf_metrics.loc[novices.index]
    perf_gamer: pd.Series = perf_metrics.loc[gamers.index]
    perf_nongamer: pd.Series = perf_metrics.loc[nongamers.index]
    perf_expert_vs_novice: tuple[str, TtestResult] = WELCH(perf_expert, perf_novice)
    perf_gamer_vs_not: tuple[str, TtestResult] = WELCH(perf_gamer, perf_nongamer)
    print_t_test(f"expert-novice perf metric", perf_expert_vs_novice)
    print_t_test(f"gamer-nongamer perf metric", perf_gamer_vs_not)
    total_expert_vs_novice = WELCH(vmt_expert + perf_expert, vmt_novice + perf_novice)
    total_gamer_vs_not = WELCH(vmt_gamer + perf_gamer, vmt_nongamer + perf_nongamer)
    print_t_test(f"expert-novice total metric", total_expert_vs_novice)
    print_t_test(f"gamer-nongamer total metric", total_gamer_vs_not)
    # most time taken for a whole trial????
    # max_trial_times
    # compare hmd movement between groups
    hmd_all: pd.DataFrame = pd.DataFrame.from_records([avg_hmd_velocity_etc(p) for p in survey.index], columns=["velocity", "acceleration", "jerk"])
    # TODO box plot these too
    hmd_expert: pd.DataFrame = hmd_all.loc[experts.index]
    hmd_novice: pd.DataFrame = hmd_all.loc[novices.index]
    hmd_gamer: pd.DataFrame = hmd_all.loc[gamers.index]
    hmd_nongamer: pd.DataFrame = hmd_all.loc[nongamers.index]
    hmd_expert_vs_novice: list[tuple[str, TtestResult]] = hmd_t_tests(hmd_expert, hmd_novice, "expert-novice")
    hmd_gamer_vs_not: list[tuple[str, TtestResult]] = hmd_t_tests(hmd_gamer, hmd_nongamer, "gamer-nongamer")
    # compare gaze info too
    gaze_all: pd.DataFrame = pd.DataFrame([collate_gaze_info(p) for p in survey.index]) # from list of dicts
    gaze_expert: pd.DataFrame = gaze_all.loc[experts.index]
    gaze_novice: pd.DataFrame = gaze_all.loc[novices.index]
    gaze_gamer: pd.DataFrame = gaze_all.loc[gamers.index]
    gaze_nongamer: pd.DataFrame = gaze_all.loc[nongamers.index]
    # let's t-test for everything idfk
    # expecting experts to have a lower tv_stomachpos gaze percent than novices, not sure about others
    gaze_expert_vs_novice: list[tuple[str, TtestResult]] = gaze_t_tests(gaze_expert, gaze_novice, "expert-novice")
    gaze_gamer_vs_not: list[tuple[str, TtestResult]] = gaze_t_tests(gaze_gamer, gaze_nongamer, "gamer-nongamer")
    print("*******NOW T-TESTING EVERY VIDEO METRIC INDIVIDUALLY*******")
    video_scores: pd.DataFrame = video_metrics.apply(individual_video_scoring, axis=1, result_type="expand", args=quartiles)
    for col in video_scores.columns:
        expert_scores: pd.Series = video_scores[col].loc[experts.index]
        novice_scores: pd.Series = video_scores[col].loc[novices.index]
        print_t_test(f"expert-novice {col}", WELCH(expert_scores, novice_scores))
    # wait successful bites points has p<0.05?
    print_t_test("expert-novice bite count", WELCH(video_metrics["Suture: Successful Bites"].loc[experts.index], video_metrics["Suture: Successful Bites"].loc[novices.index]))
    # confusion matrix for full cinch?
    vme: pd.DataFrame = video_metrics.loc[experts.index]
    vmn: pd.DataFrame = video_metrics.loc[novices.index]
    fc_expert_cinched: int = len(vme[vme["Suture: Full Cinch"] == "yes"])
    fc_expert_uncinched: int = len(vme[vme["Suture: Full Cinch"] == "no"])
    fc_novice_cinched: int = len(vmn[vmn["Suture: Full Cinch"] == "yes"])
    fc_novice_uncinched: int = len(vmn[vmn["Suture: Full Cinch"] == "no"])
    print(f"{fc_expert_cinched} experts cinched, {fc_expert_uncinched} couldn't, {fc_novice_cinched} novices cinched, {fc_novice_uncinched} couldn't")
    return survey

if __name__ == "__main__":
    import sys
    main(sys.argv)