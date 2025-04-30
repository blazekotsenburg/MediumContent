# 🔴 Breaking the Bot: Red Teaming LLMs with Microsoft's PyRit

This repo contains a working demo of how to orchestrate an automated adversarial attack on an LLM using [PyRit](https://github.com/microsoft/pyrit), a red teaming framework developed by Microsoft’s AI Red Team. The attack is run against a CTF-style challenge on the [Crucible platform](https://platform.dreadnode.io), where the goal is to extract a hidden flag from a target model.

## 🧠 What’s Included
- Orchestrated multi-turn LLM attack using PyRit
- Integration with OpenAI and Crucible
- Substring-based scoring logic
- `.env`-based configuration
- Example prompt objective and attack strategy

## 🚀 Setup

1. **Clone the repo** and create a Python virtual environment.
2. **Install dependencies:**

   ```bash
   pip install pyrit==0.8.1
   pip install python-dotenv==1.1.0
   ```

3. **Set up your `.env` file:**

   ```env
   CRUCIBLE_API_KEY=<your_crucible_key>
   CRUCIBLE_URL=https://platform.dreadnode.io
   CHALLENGE_URL=https://bear4.platform.dreadnode.io

   OPENAI_CHAT_MODEL=gpt-4o-mini-2024-07-18
   OPENAI_CHAT_KEY=<your_openai_key>
   OPENAI_CHAT_ENDPOINT=https://api.openai.com/v1/chat/completions
   ```

4. **Run the attack:**

   ```python
   await red_team_orchestrator.run_attack_async(objective=conversation_objective)
   ```

5. **Reset memory when done:**

   ```python
   from pyrit.memory import CentralMemory
   memory = CentralMemory.get_memory_instance()
   memory.dispose_engine()
   ```

## ⚠️ Notes

- You may hit rate limits on free OpenAI tiers (3 RPM). Consider adding billing.
- Some Crucible challenges may not be solvable using LLMs alone.
- This demo uses an in-memory database; production use may involve Azure SQL or DuckDB.

## 📚 Resources

- [PyRit GitHub](https://github.com/microsoft/pyrit)
- [Crucible by Dreadnode](https://platform.dreadnode.io)
- [PyRit Website](https://pyrit.dev)
- [Automation of AI Red Teaming (MS Research)](https://www.microsoft.com/en-us/research/publication/the-automation-advantage-of-ai-red-teaming/)
