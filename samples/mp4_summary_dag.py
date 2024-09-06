"""
Example Airflow DAG for Google Vertex AI Generative Model.
"""

from datetime import datetime, timedelta
from airflow import models
from airflow.providers.google.cloud.operators.vertex_ai.generative_model import (
    GenerativeModelGenerateContentOperator
)
from airflow.providers.google.cloud.sensors.gcs import (
    GCSObjectExistenceSensor
)

PROJECT_ID = "your-project"
LOCATION = "us-central1"
BUCKET = "your-bucket"
FILE_PATH = "videos/pixel8.mp4"
BASE_PROMPT = """
Provide a description of the video.
The description should also contain anything important which people say in the video.
"""

#-------------------------
# Begin DAG Generation
#-------------------------
with models.DAG(
    f"mp4_summary_dag",
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

    from vertexai.generative_models import Part

    gcs_object_exists_task = GCSObjectExistenceSensor(
        bucket=BUCKET, object=FILE_PATH, task_id="gcs_object_exists_task"
    )

    describe_video = GenerativeModelGenerateContentOperator(
        task_id="describe_video",
        project_id="your-project",
        location="us-central1",
        contents= [
            BASE_PROMPT,
            Part.from_uri(uri=f"gs://{BUCKET}/{FILE_PATH}", mime_type="video/mp4")

        ],
        pretrained_model="gemini-1.5-flash-001"
    )

    generate_tabular_data = GenerativeModelGenerateContentOperator(
        task_id="generate_tabular_data",
        project_id="your-project",
        location="us-central1",
        contents= [
            "Generate 10 rows of tabular data for this video.",
            Part.from_uri(fileUri=f"gs://{BUCKET}/{FILE_PATH}", mime_type="video/mp4")

        ],
        pretrained_model="gemini-1.5-flash-001"
    )

    gcs_object_exists_task >> describe_video >> generate_tabular_data