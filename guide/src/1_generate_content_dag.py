"""
Example Airflow DAG for Google Vertex AI Generative AI.
"""

from datetime import datetime, timedelta

from vertexai.generative_models import HarmBlockThreshold, HarmCategory, Tool, grounding

from airflow import models
from airflow.providers.google.cloud.operators.vertex_ai.generative_model import (
    GenerativeModelGenerateContentOperator,
)

# --------------------------------------------------------------------------------------------------
# Constants / input
# --------------------------------------------------------------------------------------------------
PROJECT_ID = "your-project"
REGION = "us-central1"

# Provides speed and efficiency for high-volume, quality, cost-effective apps.
# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-1.5-flash
FLASH_MODEL = "gemini-1.5-flash"

# Supports text or chat prompts for a text or code response. Supports long-context understanding up to the maximum input token limit.
# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-1.5-pro
PRO_MODEL = "gemini-1.5-pro-001"

# System instructions are like a preamble that you add before the LLM gets exposed to any further
# instructions from the user. It lets users steer the behavior of the model based on their specific
# needs and use cases. When you set a system instruction, you give the model additional context to
# understand the task, provide more customized responses, and adhere to specific guidelines over
# the full user interaction with the model. For developers, product-level behavior can be specified
# in system instructions, separate from prompts provided by end users
# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/system-instructions
SYSTEM_INSTRUCTION = "Always respond in plain-text only."

# Per request settings for blocking unsafe content.
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# Optional. A piece of code that enables the system to interact with external systems to perform an
# action, or set of actions, outside of knowledge and scope of the model.
# https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview
TOOLS = [Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())]

# top_k: Specify a lower value for less random responses and a higher value for more random
#        responses. (1-40) default: model dependent.
# top_p: Specify a lower value for less random responses and a higher value for more random
#        responses. (0.0 - 1.0) default: model dependent.
# temperature: Temperature controls the degree of randomness in token selection. Lower temperatures
#              are good for prompts that require a less open-ended or creative response, while
#              higher temperatures can lead to more diverse or creative results.
DETERMINISTIC_GEN_CONFIG = {"top_k": 1, "top_p": 0.0, "temperature": 0.1}
RANDOM_GEN_CONFIG = {"top_k": 40, "top_p": 0.9, "temperature": 1.0}
BALANCED_GEN_CONFIG = {"top_k": 20, "top_p": 0.5, "temperature": 0.5}

SAMPLE_PROMPT = """
Summarize the following article. Article: To make a classic spaghetti carbonara, start by 
bringing a large pot of salted water to a boil. While the water is heating up, cook pancetta or 
guanciale in a skillet with olive oil over medium heat until it's crispy and golden brown. Once 
the pancetta is done, remove it from the skillet and set it aside. In the same skillet, whisk 
together eggs, grated Parmesan cheese, and black pepper to make the sauce. When the pasta is 
cooked al dente, drain it and immediately toss it in the skillet with the egg mixture, adding a 
splash of the pasta cooking water to create a creamy sauce.
"""

# --------------------------------------------------------------------------------------------------
# Begin DAG generation
# --------------------------------------------------------------------------------------------------

with models.DAG(
    "generate_content_dag_v1",
    description="Demonstration of Vertex AI Generative AI on Airflow/Composer",
    tags=[
        "demo",
        "vertex_ai",
        "generative_ai",
    ],
    schedule="@once",
    catchup=False,
    is_paused_upon_creation=True,
    max_active_runs=1,
    default_args={
        "start_date": datetime(2024, 1, 1),
        "owner": "Google",
        "depends_on_past": False,
        "retries": 0,
        "retry_delay": timedelta(minutes=1),
        "sla": timedelta(minutes=55),
        "execution_timeout": timedelta(minutes=60),
    },
) as dag:
    # ----------------------------------------------------------------------------------------------
    # Submit prompt to various model configurations
    # ----------------------------------------------------------------------------------------------

    # The Gemini model family includes models that work with multimodal prompt requests. The term
    # multimodal indicates that you can use more than one modality, or type of input, in a prompt.
    # Models that aren't multimodal accept prompts only with text. Modalities can include text,
    # audio, video, and more. Generate a non-streaming model response from a text input.
    # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
    pro_model_task = GenerativeModelGenerateContentOperator(
        task_id="pro_model_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=PRO_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        contents=[SAMPLE_PROMPT],
    )

    flash_model_task = GenerativeModelGenerateContentOperator(
        task_id="flash_model_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=FLASH_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        contents=[SAMPLE_PROMPT],
    )

    # ultra_model_task = GenerativeModelGenerateContentOperator(
    #     task_id="ultra_model_task",
    #     project_id=PROJECT_ID,
    #     location=REGION,
    #     pretrained_model=ULTRA_MODEL,
    #     contents=[SAMPLE_PROMPT],
    # )

    deterministic_pro_model_task = GenerativeModelGenerateContentOperator(
        task_id="deterministic_pro_model_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=PRO_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        contents=[SAMPLE_PROMPT],
        generation_config=DETERMINISTIC_GEN_CONFIG,
        safety_settings=SAFETY_SETTINGS,
    )

    balanced_pro_model_task = GenerativeModelGenerateContentOperator(
        task_id="balanced_pro_model_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=PRO_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        contents=[SAMPLE_PROMPT],
        generation_config=BALANCED_GEN_CONFIG,
        safety_settings=SAFETY_SETTINGS,
    )

    random_pro_model_task = GenerativeModelGenerateContentOperator(
        task_id="random_pro_model_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=PRO_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        contents=[SAMPLE_PROMPT],
        generation_config=RANDOM_GEN_CONFIG,
        safety_settings=SAFETY_SETTINGS,
    )

    grounded_pro_model_task = GenerativeModelGenerateContentOperator(
        task_id="grounded_model_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=PRO_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        contents=[SAMPLE_PROMPT],
        tools=TOOLS,
        safety_settings=SAFETY_SETTINGS,
    )

    # ----------------------------------------------------------------------------------------------
    # Dependencies
    # ----------------------------------------------------------------------------------------------
    [
        flash_model_task,
        pro_model_task,
        deterministic_pro_model_task,
        balanced_pro_model_task,
        random_pro_model_task,
        grounded_pro_model_task,
    ]
