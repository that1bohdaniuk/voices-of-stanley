import asyncio
from memory.archive import *
from misc import test_events_gen
from misc.test_events_gen import generate_test_events


async def main():
    chroma_client = await initialize_chroma_client()
    collection = chroma_client.get_collection("events-collection")
    #events: List[GameEventModel] = generate_test_events(1000)

    #await embed_bunch(events)
    _event, = generate_test_events(1)
    result = await retrieve(_event)
    print(len(result), result, sep='\n')



if __name__ == "__main__":
    asyncio.run(main())