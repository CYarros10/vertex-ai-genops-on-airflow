"""
Example Airflow DAG for Google Vertex AI Generative Model.
"""

from datetime import datetime, timedelta

from airflow import models
from airflow.providers.google.cloud.sensors.gcs import (
    GCSObjectExistenceSensor
)
from airflow.providers.google.cloud.operators.vertex_ai.generative_model import (
    GenerativeModelGenerateContentOperator
)
from airflow.operators.python import (
    PythonOperator
)
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import (
    GCSToBigQueryOperator,
)

PROJECT_ID = "your-project"
LOCATION = "us-central1"

BASE_PROMPT = """
You will be given a customer service transcript.
Your job is to analyze the transcript and provide feedback on the customer service agent.
Create JSON data with the following fields ('personalization', 'competency', 'convenience', 'proactiveness', and 'reasoning').
Values for 'personalization', 'competency', 'convenience', and 'proactiveness' on a scale of 1 to 10.
Values for 'reasoning' should include a few sentences explaining why you gave the scores.
Keep your response on a single line without formatting.
"""

#-------------------------
# Begin DAG Generation
#-------------------------
with models.DAG(
    f"customer_service_feedback_dag",
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

    def write_to_gcs(bucket_name, blob_name, data, content_type=None):
        """
        Writes data to a Google Cloud Storage bucket.

        Args:
            bucket_name: The name of the GCS bucket.
            blob_name: The name of the blob (file) to create or overwrite.
            data: The data to write, either as bytes or a string.
            content_type: (Optional) The MIME type of the data (e.g., 'text/plain', 'image/jpeg').
        """
        from google.cloud import storage

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        data.replace("```json", "")
        data.replace("```", "")

        blob.upload_from_string(data)

        print(f"Data written to gs://{bucket_name}/{blob_name}")


    gcs_object_exists_task = GCSObjectExistenceSensor(
        bucket="your-bucket", object="sample_data/transcript1.txt", task_id="gcs_object_exists_task"
    )

    generate_feedback_task = GenerativeModelGenerateContentOperator(
        task_id="generate_feedback_task",
        project_id="your-project",
        location="us-central1",
        contents= [
            BASE_PROMPT,
            Part.from_uri(uri="gs://your-bucket/sample_data/transcript1.txt", mime_type="text/plain"),
        ],
        generation_config = {
            "temperature": 0.0
        },
        pretrained_model="gemini-1.5-flash-001"
    )

    write_to_gcs_task = PythonOperator(
        task_id=f"write_to_gcs_task",
        python_callable=write_to_gcs,
        op_kwargs={
            "bucket_name": "your-bucket",
            "blob_name": "sample_data/transcript_feedback/sample.json",
            "data": "{{ ti.xcom_pull(task_ids='generate_feedback_task') }}"
        },
    )

    load_to_bigquery_task = GCSToBigQueryOperator( 
        task_id="load_to_bigquery",
        bucket="your-bucket",
        source_objects=["sample_data/transcript_feedback/sample.json"],  
        destination_project_dataset_table='your-project.sandbox.customerservice',
        source_format='NEWLINE_DELIMITED_JSON',
        create_disposition='CREATE_IF_NEEDED',
        write_disposition='WRITE_APPEND',
    )

    gcs_object_exists_task >> generate_feedback_task >> write_to_gcs_task >> load_to_bigquery_task