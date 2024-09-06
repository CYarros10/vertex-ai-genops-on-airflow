"""
"""

from datetime import datetime, timedelta
from vertexai.generative_models import Part

from airflow import models
from airflow.providers.google.cloud.transfers.postgres_to_gcs import (
    PostgresToGCSOperator,
)
from airflow.providers.google.cloud.sensors.gcs import (
    GCSObjectsWithPrefixExistenceSensor,
)
from airflow.providers.google.cloud.operators.vertex_ai.generative_model import (
    GenerativeModelGenerateContentOperator
)


formatted_date = datetime.today().strftime("%Y %m %d")
GCS_BUCKET = "your-bucket"
GCS_OBJECT_PATH = f"composer-daily-reports/{formatted_date}"

POSTGRES_CONNECTION_ID = "airflow_db"
FILE_FORMAT = "csv"

default_args = {
    "owner": "auditing",
    "depends_on_past": False,
    "email": [""],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=30)
}


with models.DAG(
    f"composer_status_report",
    tags=["airflow_db"],
    description=f"Export contents of airflow metadata DB to GCS location and analyze with Gemini.",
    is_paused_upon_creation=True,
    catchup=False,
    start_date=datetime(2024, 1, 1),
    max_active_runs=1,
    default_args=default_args,
    schedule_interval="@once", # change to daily
) as dag:

    export_dag_run = PostgresToGCSOperator(
        task_id="export_dag_run",
        postgres_conn_id=POSTGRES_CONNECTION_ID,
        sql="SELECT * FROM dag_run WHERE execution_date > NOW() - INTERVAL '1 day';",
        bucket=GCS_BUCKET,
        filename=f"{GCS_OBJECT_PATH}/dag_run/dag_run.{FILE_FORMAT}",
        export_format=FILE_FORMAT,
        field_delimiter=",",
        gzip=False,
        use_server_side_cursor=False,
        execution_timeout=timedelta(minutes=15),
    )

    # will fail if table row count == 0 (no file will exist)
    output_check = GCSObjectsWithPrefixExistenceSensor(
        task_id="output_check",
        bucket=GCS_BUCKET,
        prefix=f"{GCS_OBJECT_PATH}/dag_run/dag_run.{FILE_FORMAT}",
        mode="reschedule",
        poke_interval=60,
        timeout=60 * 3,
    )

    dag_run_report = GenerativeModelGenerateContentOperator(
        task_id="dag_run_report",
        project_id="your-project",
        location="us-central1",
        contents= [
            """
            You are an Airflow Metadata Database expert. 
            You are tasked with providing today's Airflow status report. 
            You will be provided the dag_run table data.
            If you see potential problems in the data, please provide recommendations to fix.
            If you see trends in the data, please describe them.

            Respond in plain-text paragraph form. Please use 100 words or less.
            """,
            Part.from_uri(uri=f"gs://{GCS_BUCKET}/{GCS_OBJECT_PATH}/dag_run/dag_run.{FILE_FORMAT}", mime_type="text/csv")

        ],
        pretrained_model="gemini-1.5-flash-001"
    )

export_dag_run >> output_check >> dag_run_report
