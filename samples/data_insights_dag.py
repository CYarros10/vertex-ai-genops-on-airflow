"""
Example Airflow DAG for Google Vertex AI Generative Model.
"""

from datetime import datetime, timedelta

from airflow import models
from airflow.utils.task_group import TaskGroup
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryInsertJobOperator,
    BigQueryGetDataOperator,
)
from airflow.providers.google.cloud.operators.vertex_ai.generative_model import (
    GenerativeModelGenerateContentOperator
)

PROJECT_ID = "your-project"
LOCATION = "us-central1"
DATASET = "sandbox"
TABLE = "geminisampledata"
RESULTS = 100

BASE_PROMPT = "You are a Data Scientist tasked with answer questions in plain-text."
BASE_PROMPT = """
You're a Data Scientist and Business Analyst.  You will be given a subset of data to analyze.

1. Respond in a plain-text paragraph only.
2. Respond in less than 100 words.
"""

#-------------------------
# Begin DAG Generation
#-------------------------
with models.DAG(
    f"data_insights_dag",
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
    
     # Run BigQueryInsertJobOperator to create new Temporary Table
    create_table = BigQueryInsertJobOperator(
        task_id="create_table",
        configuration={
            "query": {
                "query": """
CREATE TABLE IF NOT EXISTS `{dataset}.{table}` AS
(
SELECT subscriber_type, bike_type, start_time, start_station_name, end_station_name, duration_minutes FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips` ORDER BY start_time DESC LIMIT {results}
)
""".format(dataset=DATASET, table=TABLE, results=RESULTS),
                "useLegacySql": False}
        },
        location="US",
    )

    # Get Data
    get_data = BigQueryGetDataOperator(
        task_id="get_data",
        dataset_id=DATASET,
        table_id=TABLE,
        use_legacy_sql=False,
        max_results=RESULTS,
        location="US",
    )

    with TaskGroup("gemini_data_insights") as gemini_data_insights:

        ask_for_trends = GenerativeModelGenerateContentOperator(
            task_id="ask_for_trends",
            project_id="your-project",
            location="us-central1",
            contents= [
                f"{BASE_PROMPT}",
                "Do you notice any underlying trends in customer activity?",
                str(get_data.output)
            ],
            pretrained_model="gemini-1.5-flash-001"
        )

        ask_for_user_experience_improvement = GenerativeModelGenerateContentOperator(
            task_id="ask_for_user_experience_improvement",
            project_id="your-project",
            location="us-central1",
            contents= [
                f"{BASE_PROMPT}",
                "What are some actions we can take to improve the user experience of our customers while still maintaining our current budget?",
                str(get_data.output)
            ],
            pretrained_model="gemini-1.5-flash-001"
        )

        ask_for_monetization_strategy = GenerativeModelGenerateContentOperator(
            task_id="ask_for_monetization_strategy",
            project_id="your-project",
            location="us-central1",
            contents= [
                f"{BASE_PROMPT}",
                "How can we further monetize our business or add new revenue streams?",
                str(get_data.output)
            ],
            pretrained_model="gemini-1.5-flash-001"
        )

        ask_for_new_data_points = GenerativeModelGenerateContentOperator(
            task_id="ask_for_new_data_points",
            project_id="your-project",
            location="us-central1",
            contents= [
                f"{BASE_PROMPT}",
                "Provide ideas for new data points to capture that will improve our data analytics.",
                str(get_data.output)
            ],
            pretrained_model="gemini-1.5-flash-001"
        )

    create_table >> get_data >> gemini_data_insights