import math
import numpy as np
print("importing pandas...")
import pandas as pd
import sys
from enum import StrEnum
from pathlib import Path
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
    CSVData.TechniqueInfo: pd.DataFrame(columns=["PatternType","SutureSetCount","ArrowPlacingTIme"]),
    CSVData.AssistantCalls: pd.DataFrame(columns=["Seconds","Command","Interpretation"]),
    CSVData.HMDPosition: pd.DataFrame(columns=["Seconds","PositionX","PositionY","PositionZ","RotationX","RotationY","RotationZ","QuatX","QuatY","QuatZ","QuatW"]),
    CSVData.GraspInfo: pd.DataFrame(columns=["GraspTime", "UngraspTime", "GraspPosition(x)", "GraspPosition(y)", "GraspPosition(z)"])
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
    comm_grasp: list[int] = [] # 0 for telling assistant to grasp for each bite
    comm_ungrasp: list[int] = [] # same for ungrasping
    comm_remove_helix: int = 1 # 0 for telling assistant to switch off helix ig? each set should go helix>ttag>cinch
    comm_use_cinch: int = 1 # 0 for telling assistant to activate cinch
    comm_drop_cinch: int = 1 # 0 for telling assistant to deploy cinch after tightening

gaze_targets: frozenset[str] = frozenset({"tv", "tv_stomachpos", "Instructions_TV", "floor"})

def main(trial: Path):
    print("loading the csvs...")
    data: dict[CSVData, pd.DataFrame] = {
        name: load_csv(trial, name)
        for name in CSVData
    }
    # TODO
    metrics = Metrics()
    # skip all insertion metrics
    # metrics 10-12 (marking, fig 4): add zones to argonparallelinfo (DONE) or the cross product thing to argonmarkinfo
    for hit in data[CSVData.ArgonParallelInfo].itertuples(name=CSVData.ArgonParallelInfo):
        # all the marking metrics start at 5, then we lower them to 3 or 0
        # also api columns aren't valid python identifiers so let's index by number
        score: int = 0 if math.hypot(hit[1], hit[2], hit[3]) <= 1.0 else 3
        if hit[4].casefold() == "anterior":
            metrics.mark_anterior = min(metrics.mark_anterior, score)
        elif hit[4].casefold() == "posterior":
            metrics.mark_posterior = min(metrics.mark_posterior, score)
        elif hit[4].casefold() == "greatercurvature":
            metrics.mark_GC = min(metrics.mark_GC, score)
    num_suture_sets: int = data[CSVData.SutureInfo]["SutureSetCount"].max()
    for i_suture_set in range(1, num_suture_sets + 1):
        # something here
        pass
    # metrics 17-23 (suturing, fig 6): compare sutureinfo to graspinfo, argon(mark|parallel)info. positions are recorded in unity units - figure out how big 0.5 cm is
    # metric 24 (fig 6): wtf is a suture direction? gap between bites?
    # metric 25 (fig 6): skip, all bites are full thickness
    # metric 26 (fig 6): just count from sutureinfo.suturecount
    # metric 27 (fig 6): techniqueinfo is supposed to have this. instruct user to do u-shaped sutures, review recorder code
    # metric 28 (fig 6): premature deployment is if ttag drops while it's not the active instrument. track that somewhere, new recorder maybe?
    # metric 29 (fig 7): note this ge junction's position, do linear algebra to figure out proximity
    # metric 30 (fig 7): last suture's stomachpart
    # metric 33 (fig 7): just count from sutureinfo.suturesetcount
    # metric 35 (fig 8): bleedingfinishtime
    # metrics 38-45 (fig 10): 40 is grasping and 41 is ungrasping, ignore 39 and 42 unless we bring back extend/retract commands, others correspond to commands fairly obviously 

def load_csv(trial: Path, csv: CSVData) -> pd.DataFrame:
    try:
        return pd.read_csv(trial / (csv + ".csv"))
    except pd.errors.EmptyDataError:
        print(f"{csv} is empty!", file=sys.stderr)
        return nodata[csv]

if __name__ == "__main__":
    main(Path(sys.argv[1]))