import numpy as np
import pandas as pd
from enum import StrEnum
from pathlib import Path

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

gaze_targets: set[str] = {"tv", "tv_stomachpos", "Instructions_TV", "floor"}

def main(trial: Path):
    pass

if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]))