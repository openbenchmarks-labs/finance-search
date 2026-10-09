import os

for k in ["EXA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
    os.environ.setdefault(k, "test")
