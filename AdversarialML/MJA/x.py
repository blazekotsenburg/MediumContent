from models.prompt import Prompt
from dotenv import load_dotenv

import os

load_dotenv(dotenv_path="/Users/blazekotsenburg/Documents/Source/Repos/MediumContent/AdversarialML/MJA/.env")
PATH_METAPHOR_SYS_PROMPT=os.getenv("PATH_METAPHOR_SYS_PROMPT")
PATH_CONTEXT_SYS_PROMPT=os.getenv("PATH_CONTEXT_SYS_PROMPT")
PATH_ADV_SYS_PROMPT=os.getenv("PATH_ADV_SYS_PROMPT")

PATH_METAPHOR_USR_PROMPT=os.getenv("PATH_METAPHOR_USR_PROMPT")
PATH_CONTEXT_USR_PROMPT=os.getenv("PATH_CONTEXT_USR_PROMPT")
PATH_ADV_USR_PROMPT=os.getenv("PATH_ADV_USR_PROMPT")

p = Prompt.load_from_file(file_path=PATH_METAPHOR_USR_PROMPT)
print(p.render(sen_content="It works"))