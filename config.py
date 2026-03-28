# stores all of parameters
import time

# interval that governs buffer flush into orchestrator clock
CLOCK_INTERVAL: int = 5
# tension threshold, і так понятно
TENSION_THRESHOLD: int = 100
# dt | Decay rate for time-weighted RAG with importance score (look up miro)
TWRAG_DECAY_RATE: float = 0.1
# minutes after which events become obsolete
EVENT_PURGE_TIME: int = 60
# threshold for importance value below which events become purge-prone
EVENT_PURGE_IMPORTANCE_THRESHOLD: float = 3.5
# current ingame time
CURRENT_TIME: float = time.time()
# unix timestamp that indicates when the last pruning was done
LAST_PRUNE_TIME: float
# threshold which governs after how many in-game days pruning proccess starts
EVENT_PRUNE_TIME_THRESHOLD_SECONDS: float = 5*24*60*60