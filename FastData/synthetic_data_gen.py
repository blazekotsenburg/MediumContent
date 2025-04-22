from fastdata.core import FastData
from datasets import load_dataset

import csv

class UserGeneratedContent:
    """User generated content containing the content and a label that classifies the content."""
    def __init__(cls, content:str, label:str):
        cls.content = content
        cls.label = label

    def __repr__(cls):
        return "{" + f"'content': {cls.content}, 'label': {cls.label}" + "}"
    
    def to_dict(cls):
        return {
            "content": cls.content,
            "label": cls.label
        }
    
def generate_synth_data(fast_data, personas, examples, policies, labels, sys_prompt):
    """
    Generate synthetic user-generated content.
    """
    prompt_template = """
    Create synthetic user-generated content that would be found on social media platforms or other online communities that either violates one of the
    below moderation policies, or is just labeled as Safe. Label the content as one of the following policy labels: <labels>{labels}</labels>.
    Use the following persona when generating content: <persona>{persona}</persona>.

    <policies>
    {policies}
    </policies>

    <examples>
    {examples}
    </examples>
    """

    synth_data = fast_data.generate(prompt_template=prompt_template, 
                                    inputs=[{"persona": persona, "examples": examples, "policies": policies, "labels": labels} for persona in personas],
                                    schema=UserGeneratedContent,
                                    sp=sys_prompt,
                                    )
    
    return [content for content in synth_data if content is not None]

def load_personas(num_personas: int = 100):
    return load_dataset("proj-persona/PersonaHub", "persona", split='train').select(range(num_personas))['persona']

examples = [UserGeneratedContent(content= "Congratulations to the both of you on your engagement! Much love and so proud!", label="Safe"),
            UserGeneratedContent(content= "You should walk into oncoming traffic", label= "Self‑harm & Suicide"),
            UserGeneratedContent(content= "Totally can meet at my house, 42 Wallaby Way, 9 o'clock", label="Privacy & Personal Data Exposure"),
            UserGeneratedContent(content= "You just won a random raffle worth up to $10,000. Please send us your email and address so we can send your money immediately!", label= "Spam & Scams")]

with open("policies.txt", 'r') as file:
    policies = file.read()
    
LABELS=["Spam & Scams", "Self‑harm & Suicide", "Safe", "Privacy & Personal Data Exposure"]

fast_data = FastData(model="claude-3-7-sonnet-20250219")
personas = load_personas()
synth_data = generate_synth_data(fast_data=fast_data, 
                                 personas=personas, 
                                 examples=examples, 
                                 policies=policies, 
                                 labels=LABELS, 
                                 sys_prompt="You're an assistant that helps generate user content.")

with open("synthetic_data.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["content", "label"])
    writer.writeheader()
    writer.writerows([item.to_dict() for item in synth_data])

