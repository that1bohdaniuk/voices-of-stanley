import uuid
import time
import random
from datetime import datetime, timedelta
from typing import List

from faker import Faker
from api.schemas import GameEventModel

fake = Faker()

def generate_test_events(num: int) -> List[GameEventModel]:
    events: List[GameEventModel] = []
    event_types = ["Radar ping", "Server crash", "Entity detected", "Signal downloaded"
                   "Health depleted","Drone arrived", "Purchase made"]
    location_types = ["Server room", "Living room", "Bedroom", "Garage", "Computer room", "Canteen", "Hallway",
                      "Outside of base", "Forest", "Satellite", "Wind turbine", "Transformer"]


    for _ in range(num):
        days_ago = random.uniform(0, 10)
        simulated_time = time.time() - (60*60*24*days_ago)

        id = uuid.uuid4()
        label =f"{random.choice(event_types)}: {fake.sentence()}"
        timestamp = simulated_time
        location = random.choice(location_types)
        importance = random.uniform(0.1, 10)
        details={"inventory": fake.word(), "hungriness": random.uniform(1, 100)}
        events.append(GameEventModel(id=id, label=label, timestamp=timestamp, location=location, importance=importance, details=details))
    return events