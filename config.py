# stores all of parameters
import time

# interval that governs buffer flush into orchestrator clock
CLOCK_INTERVAL: int = 5
# tension threshold, і так понятно
TENSION_SUM_THRESHOLD: float = 100
# dt | Decay rate for time-weighted RAG with importance score (look up miro)
TWRAG_DECAY_RATE: float = 0.1
# minutes after which events become obsolete
EVENT_PURGE_TIME: int = 7*75
# threshold for importance value below which events become purge-prone
EVENT_PURGE_IMPORTANCE_THRESHOLD: float = 3.5
# unix timestamp that indicates when the last pruning was done
LAST_PRUNE_TIME: float
# threshold which governs after how many in-game days pruning proccess starts
EVENT_PRUNE_TIME_THRESHOLD_SECONDS: float = 5*75*60
# how many similar events archive.twrag() returns
TWRAG_RETRIEVAL_BROADNESS:int = 5
# model name for miner
MINER_MODEL: str = 'miner-9B'
# model name for pruner
PRUNER_MODEL: str = 'pruner-4B'
# model name for director
DIRECTOR_MODEL: str = 'director-9B'
# importance bound that signals to director to react immediately
FORCE_DIRECTOR_THRESHOLD: float = 8.0
# amount of routine tasks player should do in a row for director to trigger
IDLE_DIRECTOR_THRESHOLD: int = 10