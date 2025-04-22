# Synthetic Data Generation for LLM Moderation with FastData

This project demonstrates how to use [FastData](https://github.com/AnswerDotAI/fastdata), an open-source Python library, to generate synthetic user-generated content (UGC) for content moderation use cases using Anthropic’s Claude models.

Whether you're building a fine-tuning dataset or prototyping a moderation classifier, synthetic data can help you quickly simulate real-world scenarios at scale—without the cost or risk of human-labeled examples.

---

## 🔧 Requirements

- Python 3.13.3
- [Anthropic API Key](https://docs.anthropic.com/claude/docs/getting-started)
- `fastdata==0.0.4`
- `datasets==3.5.0`

Install dependencies:

```bash
pip install python-fastdata==0.0.4 datasets==3.5.0
```

Export your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

---

## 🎯 Project Goal

Generate a CSV dataset of synthetic user messages that simulate real online behavior and classify them according to moderation policy labels:

- `Spam & Scams`
- `Self-harm & Suicide`
- `Privacy & Personal Data Exposure`
- `Safe`

The output CSV will have two columns:  
**`content`** (user-generated text) and **`label`** (policy classification).

---

## 📁 Dataset Uniqueness

To avoid duplication and bias, the script loads diverse persona samples from the [HuggingFace `proj-persona/PersonaHub`](https://huggingface.co/datasets/proj-persona/PersonaHub) dataset. Each sample is generated with a unique persona for realistic diversity.

---

## ⚙️ How It Works

1. **Define a Schema** using a simple Python class  
2. **Load Personas** using HuggingFace’s datasets library  
3. **Generate Samples** using `FastData.generate()` with dynamic prompt templating  
4. **Export to CSV** for downstream use

---

## 🧪 Example Output

Running the script will generate `synthetic_data.csv`, containing content like:

```csv
content,label
"You should walk into oncoming traffic","Self-harm & Suicide"
"Congrats on your promotion! Well deserved.","Safe"
"Send your email to claim your $10,000 prize","Spam & Scams"
```

---

## 🧠 Things to Consider

- FastData **only supports Anthropic models** for now  
- Consider using a secondary LLM or human feedback to **evaluate quality**  
- Large-scale generation can incur **API costs and rate limits**  
- **Statistical summaries** (label balance, duplication rate, etc.) can improve dataset trustworthiness

---

## 📚 Resources

- [FastData GitHub Repo](https://github.com/AnswerDotAI/fastdata)  
- [Medium Walkthrough](https://medium.com/@blazekotsenburg) (this repo is based on the article)  
- [HuggingFace Persona Dataset](https://huggingface.co/datasets/proj-persona/PersonaHub)  
- [Anthropic API Docs](https://docs.anthropic.com/claude)

---

## 📤 Contributions

Feel free to fork and modify the dataset schema, prompt structure, or evaluation workflow for your own use case. PRs and suggestions welcome!
