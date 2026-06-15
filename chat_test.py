from app.chat import chat

history = []
r1 = chat(history, "Which state's law governs the Chase Affiliate Agreement?")
print(f"[{r1['route']}] {r1['answer']}\n")

history += [{"role": "user", "text": "Which state's law governs the Chase Affiliate Agreement?"},
            {"role": "model", "text": r1["answer"]}]

r2 = chat(history, "and can either party terminate it without cause?")
print(f"rewritten: {r2['standalone_question']}")
print(f"[{r2['route']}] {r2['answer']}\n")

r3 = chat(history, "thanks, you're really helpful!")
print(f"[{r3['route']}] {r3['answer']}")
