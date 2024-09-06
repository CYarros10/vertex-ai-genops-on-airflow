"""
Example Airflow DAG for Google Vertex AI Generative Model.
"""

from datetime import datetime, timedelta
from airflow import models
from airflow.providers.google.cloud.operators.vertex_ai.generative_model import (
    GenerativeModelGenerateContentOperator
)

#-------------------------
# Begin DAG Generation
#-------------------------
with models.DAG(
    f"ask_google_questions_2",
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
        "execution_timeout": timedelta(minutes=60)
    },
) as dag:

    from vertexai.generative_models import Tool
    import vertexai.preview.generative_models as generative_models


    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    ask_google_questions = GenerativeModelGenerateContentOperator(
        task_id="ask_google_questions",
        project_id="your-project",
        location="us-central1",
        contents= [
            """
            When is the next total solar eclipse in US?
            """,
        ],
        tools=[
            Tool.from_google_search_retrieval(generative_models.grounding.GoogleSearchRetrieval())
        ],
        system_instruction="Sound like a Science Professor.",
        safety_settings=safety_settings,
        generation_config={
            "temperature": 0.0
        },
        pretrained_model="gemini-1.5-flash-001"
    )

    # Requires enterprise edition connected app.
    ask_sample_datastore = GenerativeModelGenerateContentOperator(
        task_id="ask_sample_datastore",
        project_id="your-project",
        location="us-central1",
        contents= [
            """
            What's in this datastore?
            """
        ],
        tools=[
            Tool.from_retrieval(
                generative_models.grounding.Retrieval(generative_models.grounding.VertexAISearch(f"projects/488712714114/locations/global/collections/default_collection/dataStores/sample-pdfs_1724689684580"))
            )
        ],
        system_instruction="Sound like a business professional.",
        safety_settings=safety_settings,
        generation_config={
            "temperature": 0.0
        },
        pretrained_model="gemini-1.5-flash-001"
    )

    ask_google_questions >> ask_sample_datastore