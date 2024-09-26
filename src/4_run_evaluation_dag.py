"""
Example Airflow DAG for Google Vertex AI Evaluation API.
"""

from datetime import datetime, timedelta
from uuid import uuid4

from vertexai.preview.evaluation import MetricPromptTemplateExamples

from airflow import models
from airflow.providers.google.cloud.operators.vertex_ai.generative_model import (
    RunEvaluationOperator,
)

# --------------------------------------------------------------------------------------------------
# Constants / input
# --------------------------------------------------------------------------------------------------

PROJECT_ID = "cy-artifacts"
REGION = "us-central1"
BUCKET_NAME = "cy-sandbox"

FLASH_MODEL = "gemini-1.5-flash"

# System instructions are like a preamble that you add before the LLM gets exposed to any further
# instructions from the user. It lets users steer the behavior of the model based on their specific
# needs and use cases. When you set a system instruction, you give the model additional context to
# understand the task, provide more customized responses, and adhere to specific guidelines over
# the full user interaction with the model. For developers, product-level behavior can be specified
# in system instructions, separate from prompts provided by end users
# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/system-instructions
SYSTEM_INSTRUCTION = "Always respond in plain-text only."

# context: Inference-time text containing all information, which can be used in the LLM response.
# instruction: Instruction used at inference time.
# reference: Golden LLM response for reference.
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/evaluation#evaluate_an_output
CONTEXT="To make a classic spaghetti carbonara, start by bringing a large pot of salted water to a boil. While the water is heating up, cook pancetta or guanciale in a skillet with olive oil over medium heat until it's crispy and golden brown. Once the pancetta is done, remove it from the skillet and set it aside. In the same skillet, whisk together eggs, grated Parmesan cheese, and black pepper to make the sauce. When the pasta is cooked al dente, drain it and immediately toss it in the skillet with the egg mixture, adding a splash of the pasta cooking water to create a creamy sauce.",
INSTRUCTION = "Summarize the following article"
REFERENCE = "The process of making spaghetti carbonara involves boiling pasta, crisping pancetta or guanciale, whisking together eggs and Parmesan cheese, and tossing everything together to create a creamy sauce."
EVAL_DATASET = {
    "context": [
        CONTEXT
    ],
    "instruction": [INSTRUCTION],
    "reference": [
        REFERENCE    
    ],
}

# Metrics include model-based metrics, such as PointwiseMetric and PairwiseMetric, and in-memory
# computed metrics, such as rouge, bleu, and tool function-call metrics. PointwiseMetric and
# PairwiseMetric are generic model-based metrics that you can customize with your own criteria.
# Because the service takes the prediction results directly from models as input, the evaluation
# service can perform both inference and subsequent evaluation on all models supported by Vertex AI.
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/evaluation

# bleu/mean: Higher is better translation/generation quality
# bleu/std: Lower is more consistent output quality
# exactMatch/mean: Higher is better accuracy
# exactMatch/std: Lower is more consistent accuracy
# groundedness/mean: Higher is better understanding of context
# groundedness/std: Lower is more consistent understanding
# instructionFollowing/mean: Higher is better adherence to instructions
# instructionFollowing/std: Lower is more consistent instruction following
# rouge1/mean: Higher is better content similarity (single words)
# rouge1/std: Lower is more consistent unigram overlap
# rouge2/mean: Higher is better content similarity & fluency (two-word phrases)
# rouge2/std: Lower is more consistent bigram overlap
# rougeLSum/mean: Higher is better sentence-level similarity
# rougeLSum/std: Lower is more consistent sentence-level similarity
# rowCount: Higher is more evaluated responses (potentially more reliable results)
# summarizationQuality/mean: Higher is better human-rated summarization quality
# summarizationQuality/std: Lower is higher agreement among human raters
# verbosity/mean: Higher is longer responses, lower is shorter responses
# verbosity/std: Lower is more consistent response lengths
METRICS = [
    MetricPromptTemplateExamples.Pointwise.SUMMARIZATION_QUALITY,
    MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
    MetricPromptTemplateExamples.Pointwise.VERBOSITY,
    MetricPromptTemplateExamples.Pointwise.INSTRUCTION_FOLLOWING,
    "exact_match",
    "bleu",
    "rouge_1",
    "rouge_2",
    "rouge_l_sum",
]
EXPERIMENT_NAME = "eval-experiment-airflow-operator"
EXPERIMENT_RUN_NAME = "eval-experiment-airflow-operator-run"
PROMPT_TEMPLATE = "{instruction}. Article: {context}. Summary:"

SAMPLE_PROMPT = f"{INSTRUCTION}. Article: {CONTEXT}"

# --------------------------------------------------------------------------------------------------
# Begin DAG generation
# --------------------------------------------------------------------------------------------------

with models.DAG(
    "run_evaluation_dag_v1",
    description="Demonstration of Vertex AI Evaluation API on Airflow/Composer",
    tags=["demo", "vertex_ai", "generative_ai", "evaluation", "genops"],
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
    # Run evaluations for various model configurations
    # ----------------------------------------------------------------------------------------------

    # The Gen AI Evaluation Service lets you evaluate your generative models, both
    # pointwise and pairwise, across several metrics, with your own criteria. You can provide
    # inference-time inputs, LLM responses and additional parameters, and the Gen AI Evaluation
    # Service returns metrics specific to the evaluation task.
    # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/evaluation#python
    evaluate_flash_model_task = RunEvaluationOperator(
        task_id="evaluate_flash_model_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=FLASH_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        eval_dataset=EVAL_DATASET,
        metrics=METRICS,
        experiment_name=EXPERIMENT_NAME,
        experiment_run_name=f"{EXPERIMENT_RUN_NAME}-{uuid4()}",
        prompt_template=PROMPT_TEMPLATE,
    )

    # ----------------------------------------------------------------------------------------------
    # Dependencies
    # ----------------------------------------------------------------------------------------------

    evaluate_flash_model_task
