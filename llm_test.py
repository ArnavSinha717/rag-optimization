from app.router import contextualize, route

history = [
    {"role": "user",  "text": "What's the governing law of the Chase Affiliate Agreement?"},
    {"role": "model", "text": "Delaware."},
]
rq = contextualize(history, "and what about termination?")
print("rewritten:", rq)
print("route:", route(rq))
print("route small-talk:", route("hi! what can you do?"))
