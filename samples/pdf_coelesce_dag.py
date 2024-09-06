"""
Example Airflow DAG for Google Vertex AI Generative Model.
"""

from datetime import datetime, timedelta
from airflow import models
from airflow.providers.google.cloud.operators.vertex_ai.generative_model import (
    GenerativeModelGenerateContentOperator
)
from airflow.operators.python import (
    PythonOperator
)


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

    if isinstance(data, str):
        data = data.encode('utf-8')  # Convert string to bytes if necessary

    blob.upload_from_string(data, content_type=content_type)

    print(f"Data written to gs://{bucket_name}/{blob_name}")

#-------------------------
# Begin DAG Generation
#-------------------------
with models.DAG(
    f"pdf_coelesce_dag",
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
    import vertexai.preview.generative_models as generative_models

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    coalesce_pdfs_task = GenerativeModelGenerateContentOperator(
        task_id="coalesce_pdfs_task",
        project_id="your-project",
        location="us-central1",
        contents= [
            """
            Coelesce all of these PDFs into a single text summary.
            Respond in plain-text paragraph form. Please use 100 words or less.
            """,
            Part.from_uri(uri="gs://your-bucket/sample_pdfs/20240819/1.pdf", mime_type="application/pdf"),
            Part.from_uri(uri="gs://your-bucket/sample_pdfs/20240819/2.pdf", mime_type="application/pdf"),
            Part.from_uri(uri="gs://your-bucket/sample_pdfs/20240819/3.pdf", mime_type="application/pdf"),
            Part.from_uri(uri="gs://your-bucket/sample_pdfs/20240819/4.pdf", mime_type="application/pdf"),
        ],
        safety_settings=safety_settings,
        pretrained_model="gemini-1.5-flash-001"
    )

    translate_summary_task = GenerativeModelGenerateContentOperator(
        task_id="translate_summary_task",
        project_id="your-project",
        location="us-central1",
        contents= [
            "Translate this summary to Spanish.",
            "{{ ti.xcom_pull(task_ids='coalesce_pdfs') }}"
        ],
        safety_settings=safety_settings,
        pretrained_model="gemini-1.5-flash-001"
    )

    write_to_gcs_task = PythonOperator(
        task_id=f"write_to_gcs_task",
        python_callable=write_to_gcs,
        op_kwargs={
            "bucket_name": "your-bucket",
            "blob_name": "sample_pdfs/20240819/coalesced/sample.txt",
            "data": "{{ ti.xcom_pull(task_ids='coalesce_pdfs') }}"
        },
    )

    coalesce_pdfs_task >> translate_summary_task >> write_to_gcs_task