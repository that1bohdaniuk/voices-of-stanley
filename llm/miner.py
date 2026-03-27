# contains system prompts and few-shot examples for miner qwen3.5:4B model
# formats the raw buffer data into discrete event chunks and assigns the 1-10 importance score
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='qwen3.5:0.8B', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object

print("\n--- Metrics ---")
print(f"Total Duration: {response['total_duration']} nanoseconds")
print(f"Load Duration: {response['load_duration']} nanoseconds")
print(f"Prompt Eval Count: {response['prompt_eval_count']} tokens")
print(f"Prompt Eval Duration: {response['prompt_eval_duration']} nanoseconds")
print(f"Eval Count: {response['eval_count']} tokens")
print(f"Eval Duration: {response['eval_duration']} nanoseconds")