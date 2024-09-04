"""
Example Airflow DAG for Google Vertex AI Count Tokens API.
"""

from datetime import datetime, timedelta

from airflow import models
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.vertex_ai.generative_model import (
    GenerativeModelGenerateContentOperator,
    CountTokensOperator,
)

# --------------------------------------------------------------------------------------------------
# Constants / input
# --------------------------------------------------------------------------------------------------

PROJECT_ID = "your-project"
REGION = "us-central1"

# Provides speed and efficiency for high-volume, quality, cost-effective apps.
# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-1.5-flash
FLASH_MODEL = "gemini-1.5-flash"

# Prompt Budgets
CHARACTER_BUDGET = 1000
TOKEN_BUDGET = 500

# System instructions are like a preamble that you add before the LLM gets exposed to any further
# instructions from the user. It lets users steer the behavior of the model based on their specific
# needs and use cases. When you set a system instruction, you give the model additional context to
# understand the task, provide more customized responses, and adhere to specific guidelines over
# the full user interaction with the model. For developers, product-level behavior can be specified
# in system instructions, separate from prompts provided by end users
# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/system-instructions
SYSTEM_INSTRUCTION = "Always respond in plain-text only."


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
    "count_tokens_dag_v1",
    description="Demonstration of Vertex AI Count Tokens API on Airflow/Composer",
    tags=["demo", "vertex_ai", "generative_ai"],
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
    # Count tokens for sample prompt and validate within budget
    # ----------------------------------------------------------------------------------------------

    # The CountTokens API calculates the number of input tokens before sending a request to the
    # Gemini API. Use the CountTokens API to prevent requests from exceeding the model context
    # window, and estimate potential costs based on billable characters.
    # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/count-tokens#python
    count_tokens_task = CountTokensOperator(
        task_id="count_tokens_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=FLASH_MODEL,
        contents=[SAMPLE_PROMPT],
    )


    def validate_tokens(character_budget, token_budget, total_billable_characters, total_tokens):
        return int(total_billable_characters) < character_budget and int(total_tokens) < token_budget

    validate_tokens_task = PythonOperator(
        task_id="validate_tokens_task",
        python_callable=validate_tokens,
        op_kwargs={
            "character_budget": CHARACTER_BUDGET,
            "token_budget": TOKEN_BUDGET,
            "total_billable_characters": "{{ task_instance.xcom_pull(task_ids='count_tokens_task', key='total_billable_characters') }}",
            "total_tokens": "{{ task_instance.xcom_pull(task_ids='count_tokens_task', key='total_tokens') }}",
        },
        provide_context=True,
    )

    # ----------------------------------------------------------------------------------------------
    # Submit prompt to various model configurations
    # ----------------------------------------------------------------------------------------------

    # The Gemini model family includes models that work with multimodal prompt requests. The term
    # multimodal indicates that you can use more than one modality, or type of input, in a prompt.
    # Models that aren't multimodal accept prompts only with text. Modalities can include text,
    # audio, video, and more. Generate a non-streaming model response from a text input.
    # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
    flash_model_task = GenerativeModelGenerateContentOperator(
        task_id="flash_model_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=FLASH_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        contents=[SAMPLE_PROMPT],
    )

    # ----------------------------------------------------------------------------------------------
    # Dependencies
    # ----------------------------------------------------------------------------------------------

    count_tokens_task >> validate_tokens_task >> flash_model_task
