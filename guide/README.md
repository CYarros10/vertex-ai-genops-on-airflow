# A Guide to Vertex AI LLMops on Airflow

## Getting Started



## 1. Generating Content

Here we will use Airflow operators to submit requests to Vertex AI / Gemini models.

**example:**
```
    pro_model_task = GenerativeModelGenerateContentOperator(
        task_id="pro_model_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=PRO_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        contents=[SAMPLE_PROMPT],
    )
```
**Sample DAG graph:**

![generate_content_image](images/generate_content.png)

[source code](src/1_generate_content_dag.py)

## 2. Enforcing Budgets

Here we will use Airflow operators to count tokens before submitting a request to Vertex AI / Gemini models, ensuring budget restraints are considered.

**example:**
```
    count_tokens_task = CountTokensOperator(
        task_id="count_tokens_task",
        project_id=PROJECT_ID,
        location=REGION,
        pretrained_model=FLASH_MODEL,
        contents=[SAMPLE_PROMPT],
    )
```
**Sample DAG graph:**

![count_token_image](images/count_tokens.png)

[source code](src/2_count_tokens_dag.py)


## 3. Tuning LLM Models

**example:**
```
    sft_train_base_task = SupervisedFineTuningTrainOperator(
        task_id="sft_train_base_task",
        project_id=PROJECT_ID,
        location=REGION,
        source_model=PRO_MODEL,
        train_dataset=f"gs://{TRAIN_DATA_BUCKET}/{TRAIN_DATA_PATH}",
    )
```

**Sample DAG graph:**

![count_token_image](images/supervised_fine_tuning.png)

[source code](src/3_supervised_fine_tuning_dag.py)

## 4. Evaluating LLM Models

**example:**
```
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
```

**Sample DAG graph:**

![evaluation_image](images/evaluation.png)

[source code](src/4_run_evaluations_dag.py)

## 5. LLMops Pipelines

Putting it all together.

**Sample DAG graph:**

![evaluation_image](images/llmops_pipeline.png)

[source code](src/5_llmops_pipeline_dag.py)

## 6. Comparing Models

Going beyond.

**Sample DAG graph:**

![evaluation_image](images/model_comparison.png)

[source code](src/6_model_comparison_dag.py)
