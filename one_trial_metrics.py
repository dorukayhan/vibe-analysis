import math
import numpy as np
print("importing pandas...")
import pandas as pd
import sys
from enum import StrEnum
from pathlib import Path
from skspatial.objects import Plane
from typing import Literal, NamedTuple

# type checked names for each recorder's file
class CSVData(StrEnum):
    FrameRates = "FrameRates"
    SutureInfo = "SutureInfo"
    BleedingInfo = "BleedingInfo"
    TechniqueInfo = "TechniqueInfo"
    ArgonMarkInfo = "ArgonMarkInfo"
    BenchmarkInfo = "BenchmarkInfo"
    ArgonParallelInfo = "ArgonParallelInfo"
    HMDPosition = "HMDPosition"
    GazeInfo = "GazeInfo"
    AssistantCalls = "AssistantCalls"
    EndoscopePosition = "EndoscopePosition"
    GraspInfo = "GraspInfo"
    PrematureTTag = "PrematureTTag"
# in case the thing with recorders truncating files happens again
nodata: dict[CSVData, pd.DataFrame] = {
    CSVData.ArgonMarkInfo: pd.DataFrame(columns=["Zone","ArgonTime","DistanceToZone","MarkPosition(x)","MarkPosition(y)","MarkPosition(z)"]),
    CSVData.ArgonParallelInfo: pd.DataFrame(columns=["CrossProduct(x)","CrossProduct(y)","CrossProduct(z)","StomachZones"]),
    CSVData.BenchmarkInfo: pd.DataFrame(columns=["SutureSetCount","SutureCount","FPS","RenderTime","SolverTime"]),
    CSVData.BleedingInfo: pd.DataFrame(columns=["BleedingType","BleedingIndex","BleedingStartTime","BleedingFinishTime","BleedingPosition(x)","BleedingPosition(y)","BleedingPosition(z)"]),
    CSVData.FrameRates: pd.DataFrame(columns=["Seconds","Frame"]),
    CSVData.GazeInfo: pd.DataFrame(columns=["Seconds","Target","Status"]),
    CSVData.HMDPosition: pd.DataFrame(columns=["Seconds","PositionX","PositionY","PositionZ","RotationX","RotationY","RotationZ","QuatX","QuatY","QuatZ","QuatW"]),
    CSVData.SutureInfo: pd.DataFrame(columns=["SutureSetCount","SutureCount","SutureIndex","SutureTime","IsAnchorExchangeDone","StomachPart","SuturePosition(x)","SuturePosition(y)","SuturePosition(z)"]),
    CSVData.TechniqueInfo: pd.DataFrame(columns=["PatternType","SutureSetCount","ArrowPlacingTime"]),
    CSVData.AssistantCalls: pd.DataFrame(columns=["Command","Interpretation","SutureSetCount","RecordingStartTime","STTStartTime","LLMStartTime","LLMEndTime"]),
    CSVData.EndoscopePosition: pd.DataFrame(columns=["Seconds","PositionX","PositionY","PositionZ","RotationX","RotationY","RotationZ","QuatX","QuatY","QuatZ","QuatW"]),
    CSVData.GraspInfo: pd.DataFrame(columns=["GraspTime","UngraspTime","StomachZone","SutureSetCount","GraspPosition(x)","GraspPosition(y)","GraspPosition(z)"]),
    CSVData.PrematureTTag: pd.DataFrame(columns=["SutureSetCount","Seconds"])
}
# metrics we're considering
class Metrics:
    mark_anterior: int = 5 # 0 for straight line, 3 for non-straight, 5 for no mark
    mark_posterior: int = 5
    mark_GC: int = 5
    per_suture_set: list[SutureMetrics] = [] # also 0 pts if there are 5 to 8 suture sets (count with len()), 3 if more than 8, 5 if less than 5
    time_taken: float = sys.float_info.max # 0 pts if in first quartile, 3 if in second, 6 if in third, 9 if in fourth

# metrics to repeat for each suture set
class SutureMetrics:
    start: int = 5 # 0 for "just proximal to incisura angularis", 5 for elsewhere
    anterior_grasp: int = 5 # 0 for within 0.5 cm of marking
    anterior: int = 5 # 0 for correctly anchor exchanging and then suturing
    GC_grasp: int = 5
    GC: int = 5
    posterior_grasp: int = 5
    posterior: int = 5
    direction: int = 5 # 0 for going from distal (intestine end?) to proximal with 1-2 cm between successive sutures
    num_bites: int = 5 # 0 for 6 or 7 (oh no) bites, 3 for >7, 5 for <6
    u_shaped: int = 5 # 0 for u-shaped suture. instruct users to do that!
    tightened: int = 5 # 0 for proper ttag deployment and cinching, 5 for accidental ttag
    line_GE_proximity: int = 5 # 0 for line coming within 1-2 cm of GE junction, 3 if closer or within 2-3 cm, 5 if further away
    end_out_of_fundus: int = 5 # 0 if the last bite isn't in fundus, 5 if it is
    did_bleed: bool = True
    severe_bleeding: int = 5 # 0 if no severe bleeding or stopped with early cinch within 60s, 5 if not stopped
    comm_use_helix: int = 1 # 0 for telling assistant to activate helix
    comm_grasp: int = 0 # 0 for telling assistant to grasp for each bite
    comm_ungrasp: int = 0 # same for ungrasping
    comm_remove_helix: int = 1 # 0 for telling assistant to switch off helix ig? each set should go helix>ttag>cinch
    comm_use_cinch: int = 1 # 0 for telling assistant to activate cinch
    comm_drop_cinch: int = 1 # 0 for telling assistant to deploy cinch after tightening

gaze_targets: frozenset[str] = frozenset({"tv", "tv_stomachpos", "Instructions_TV", "floor"})
incisura_angularis: Plane = Plane([12.87477, 1.29898, -0.5905], [-0.5038502, -0.04318409, -0.8627108])
max_incisura_angularis_distance: float = float("nan")
one_cm: float = float("nan")

def main(trial: Path) -> Metrics:
    print("loading the csvs...")
    data: dict[CSVData, pd.DataFrame] = {
        name: load_csv(trial, name)
        for name in CSVData
    }
    metrics = Metrics()
    # skip all insertion metrics
    # metrics 10-12 (marking, fig 4): add zones to argonparallelinfo (DONE) or the cross product thing to argonmarkinfo
    for hit in data[CSVData.ArgonParallelInfo].itertuples(name=CSVData.ArgonParallelInfo):
        # all the marking metrics start at 5, then we lower them to 3 or 0
        # also api columns aren't valid python identifiers so let's index by number
        score: int = 0 if math.hypot(hit[1], hit[2], hit[3]) <= 1.0 else 3
        if hit[4] == "Anterior":
            metrics.mark_anterior = min(metrics.mark_anterior, score)
        elif hit[4] == "Posterior":
            metrics.mark_posterior = min(metrics.mark_posterior, score)
        elif hit[4] == "GreaterCurvature":
            metrics.mark_GC = min(metrics.mark_GC, score)
    metrics.time_taken = data[CSVData.FrameRates]["Seconds"].max()
    num_suture_sets: int = data[CSVData.SutureInfo]["SutureSetCount"].max()
    for i_suture_set in range(num_suture_sets):
        this_set = SutureMetrics() # TODO
        these_bites: pd.DataFrame = data[CSVData.SutureInfo].loc[data[CSVData.SutureInfo]["SutureSetCount"] == (i_suture_set+1)].sort_values(["SutureCount"])
        these_grasps: pd.DataFrame = data[CSVData.GraspInfo].loc[data[CSVData.GraspInfo]["SutureSetCount"] == (i_suture_set+1)]
        these_calls: pd.DataFrame = data[CSVData.AssistantCalls].loc[data[CSVData.AssistantCalls]["SutureSetCount"] == (i_suture_set+1)]
        # metrics 17-23 (suturing, fig 6): compare sutureinfo to graspinfo, argon(mark|parallel)info. positions are recorded in unity units - figure out how big 0.5 cm is
        # metric 17
        first_pos = these_bites[["SuturePosition(x)","SuturePosition(y)","SuturePosition(z)"]].iloc[0]
        this_set.start = 5 if incisura_angularis.distance_point(first_pos) > MAX_INCISURA_ANGULARIS_DISTANCE else 0
        # metrics 18-19
        anterior_grasps: pd.DataFrame = these_grasps.loc[these_grasps["StomachZone"] == "Anterior"]
        anterior_bites: pd.DataFrame = these_bites.loc[these_bites["StomachPart"] == "Anterior"]
        this_set.anterior_grasp = grasps_close_enough(anterior_grasps, data[CSVData.ArgonMarkInfo])
        this_set.anterior = 5 if anterior_bites.empty else 0
        # metrics 20-21
        GC_grasps: pd.DataFrame = these_grasps.loc[these_grasps["StomachZone"] == "GreaterCurvature"]
        GC_bites: pd.DataFrame = these_bites.loc[these_bites["StomachPart"] == "GreaterCurvature"]
        this_set.GC_grasp = grasps_close_enough(GC_grasps, data[CSVData.ArgonMarkInfo])
        this_set.GC = 5 if GC_bites.empty else 0
        # metrics 22-23
        posterior_grasps: pd.DataFrame = these_grasps.loc[these_grasps["StomachZone"] == "Posterior"]
        posterior_bites: pd.DataFrame = these_bites.loc[these_bites["StomachPart"] == "Posterior"]
        this_set.posterior_grasp = grasps_close_enough(posterior_grasps, data[CSVData.ArgonMarkInfo])
        this_set.posterior = 5 if posterior_bites.empty else 0
        # metric 24 (fig 6): incisura angularis plane points in general directino of proxima, let's take its normal as proximal direction
        this_set.direction = 0
        for i in range(len(these_bites)-1):
            fore: pd.Series = these_bites.iloc[i]
            aft: pd.Series = these_bites.iloc[i+1]
            direction: tuple[float, float, float] = (aft["SuturePosition(x)"] - fore["SuturePosition(x)"],
                                                     aft["SuturePosition(y)"] - fore["SuturePosition(y)"],
                                                     aft["SuturePosition(z)"] - fore["SuturePosition(z)"])
            # must point vaguely in proximal direction and have 1-2 cm distance 
            if incisura_angularis.normal.scalar_projection(direction) <= 0 or math.hypot(*direction) < one_cm or math.hypot(*direction) > 2*one_cm:
                this_set.direction = 5
                break
        # metric 25 (fig 6): skip, all bites are full thickness
        # metric 26 (fig 6): just count from sutureinfo.suturecount
        this_set.num_bites = 5 if len(these_bites) < 6 else (3 if len(these_bites) > 7 else 0)
        # metric 27 (fig 6): techniqueinfo is supposed to have this. instruct user to do u-shaped sutures, review recorder code
        this_set.u_shaped = 0 if data[CSVData.TechniqueInfo]["PatternType"].iloc[i_suture_set] == "U" else 5
        # metric 28 (fig 6): premature deployment is if ttag drops while it's not the active instrument. track that somewhere, new recorder maybe? ok new recorder
        bad_ttags: pd.DataFrame = data[CSVData.PrematureTTag].loc[data[CSVData.PrematureTTag]["SutureSetCount"] == i_suture_set+1]
        this_set.tightened = 0 if len(bad_ttags) == 0 else 5
        # metric 29 (fig 7): note this ge junction's position, do linear algebra to figure out proximity ACTUALLY ASK MARK
        # metric 30 (fig 7): last suture's stomachpart
        this_set.end_out_of_fundus = 0 if these_bites.iloc[len(these_bites)-1]["StomachPart"] != "Fundus" else 5
        # metric 33 (fig 7): just count from sutureinfo.suturesetcount
        # metric 35 (fig 8): bleedingfinishtime
        # bleeding only starts with a bite, which only happens between a grasp and ungrasp. so what if i...
        brisk_bleeds: pd.DataFrame = data[CSVData.BleedingInfo].loc[data[CSVData.BleedingInfo]["BleedingType"] == "Brisk"]
        btimes: pd.Series = brisk_bleeds["BleedingStartTime"]
        our_bbs: pd.DataFrame = brisk_bleeds.loc[these_grasps["GraspTime"].min() <= btimes <= these_grasps["UngraspTime"].max()]
        this_set.did_bleed = len(our_bbs) > 0
        if this_set.did_bleed:
            this_set.severe_bleeding = 0 if ((our_bbs["BleedingFinishTime"] - our_bbs["BleedingStartTime"]) <= 60).all() else 5
        # metrics 38-45 (fig 10): 40 is grasping and 41 is ungrasping, ignore 39 and 42 unless we bring back extend/retract commands, others correspond to commands fairly obviously
        this_set.comm_use_helix = 0 if len(these_calls[these_calls["Interpretation"] == "ActivateSpring"]) > 0 else 1
        this_set.comm_grasp = max(len(these_calls.loc[these_calls["Interpretation"] == "GraspTissue"]) - len(these_bites), 0)
        this_set.comm_ungrasp = max(len(these_calls.loc[these_calls["Interpretation"] == "UngraspTissue"]) - len(these_bites), 0)
        this_set.comm_remove_helix = 0 if len(these_calls[these_calls["Interpretation"].str.contains("Activate(TTag|Cinch)")]) > 0 else 1
        this_set.comm_use_cinch = 0 if len(these_calls[these_calls["Interpretation"] == "ActivateCinch"]) > 0 else 1
        this_set.comm_drop_cinch = 0 if len(these_calls[these_calls["Interpretation"] == "DeployCinch"]) > 0 else 1
        metrics.per_suture_set.append(this_set)
    return metrics

def load_csv(trial: Path, csv: CSVData) -> pd.DataFrame:
    try:
        return pd.read_csv(trial / (csv + ".csv"))
    except pd.errors.EmptyDataError:
        print(f"{csv} is empty!", file=sys.stderr)
        return nodata[csv]

def grasps_close_enough(grasps: pd.DataFrame, marking: pd.DataFrame) -> int:
    for grasp in grasps.itertuples(): # index, grasptime, ungrasptime, stomachzone, suturesetcount, graspposition{x,y,z}
        for mark in marking.itertuples(): # index, zone, argontime, distancetozone, markposition{x,y,z}
            if math.hypot(mark[4]-grasp[5], mark[5]-grasp[6], mark[6]-grasp[7]) <= 0.5 * one_cm:
                return 0
    return 5

if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]))))