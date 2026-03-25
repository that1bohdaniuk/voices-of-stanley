# stores all of parameters

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