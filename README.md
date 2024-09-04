# llmops-on-airflow
Step-by-step lab/guide to getting started with Google Vertex AI LLMops on Apache Airflow.

# What is LLMops?

LLMOps, or large language model operations, refers to the practices and processes involved in managing and operating large language models (LLMs). LLMs are artificial intelligence (AI) models trained on vast datasets of text and code, enabling them to perform various language-related tasks, such as text generation, translation, and question answering.

LLMOps involves a comprehensive set of activities, including:

- Model deployment and maintenance: deploying and managing LLMs on cloud platforms or on-premises infrastructure
- Data management: curating and preparing training data, as well as monitoring and maintaining data quality
- Model training and fine-tuning: training and refining LLMs to improve their performance on specific tasks
- Monitoring and evaluation: tracking LLM performance, identifying errors, and optimizing models
- Security and compliance: ensuring the security and regulatory compliance of LLM operations

LLMOps involves a number of different steps, including:

- Data collection and preparation: LLMs require large amounts of data to train. This data must be collected and prepared in a way that is suitable for training the model.
- Model development: LLMs are developed using a variety of techniques, including unsupervised learning, supervised learning, and reinforcement learning.
- Model deployment: Once a LLM has been developed, it must be deployed to a production environment. This involves setting up the necessary infrastructure and configuring the model to run on a specific platform.
- Model management: LLMs require ongoing management to ensure that they are performing as expected. This includes monitoring the model's performance, retraining the model as needed, and making sure that the model is secure.

Learn more: [Google Cloud - What is LLMOps](https://cloud.google.com/discover/what-is-llmops?hl=en)

# Generative Models

Customize and deploy Gemini models to production in Vertex AI. Gemini, a multimodal model from Google DeepMind, is capable of understanding virtually any input, combining different types of information, and generating almost any output. Prompt and test Gemini in Vertex AI using text, images, video, or code. With Gemini’s advanced reasoning and generation capabilities, developers can try sample prompts for extracting text from images, converting image text to JSON, and even generate answers about uploaded images.

Airflow provides [GenerativeModelGenerateContentOperator](https://github.com/apache/airflow/blob/d5467d6818ce7f54abd1a7a84c30f321f63405c5/airflow/providers/google/cloud/operators/vertex_ai/generative_model.py#L507) to interact with Google Generative Models.

Learn more: [Google Cloud - Generate content with the Gemini API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference)

# Tokens

The CountTokens API calculates the number of input tokens before sending a request to the Gemini API. Use the CountTokens API to prevent requests from exceeding the model context window, and estimate potential costs based on billable characters. The CountTokens API can use the same contents parameter as Gemini API inference requests.

Airflow provides [CountTokensOperator](https://github.com/apache/airflow/blob/d5467d6818ce7f54abd1a7a84c30f321f63405c5/airflow/providers/google/cloud/operators/vertex_ai/generative_model.py#L672) to interact with the Vertex AI Count Tokens API.

Learn more: [Google Cloud Vertex AI Count Tokens API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/count-tokens?hl=en)

# Tuning

Supervised fine-tuning is a good option when you have a well-defined task with available labeled data. It's particularly effective for domain-specific applications where the language or content significantly differs from the data the large model was originally trained on.

Supervised fine-tuning adapts model behavior with a labeled dataset. This process adjusts the model's weights to minimize the difference between its predictions and the actual labels.

Airflow provides [SupervisedFineTuningTrainOperator](https://github.com/apache/airflow/blob/d5467d6818ce7f54abd1a7a84c30f321f63405c5/airflow/providers/google/cloud/operators/vertex_ai/generative_model.py#L582) to interact with the Vertex AI Tuning API.

Learn more: [Google Cloud Vertex AI Tuning API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning)

# Evaluations

The Gen AI Evaluation Service lets you evaluate your large language models (LLMs), both pointwise and pairwise, across several metrics, with your own criteria. You can provide inference-time inputs, LLM responses and additional parameters, and the Gen AI Evaluation Service returns metrics specific to the evaluation task.

Metrics include model-based metrics, such as PointwiseMetric and PairwiseMetric, and in-memory computed metrics, such as rouge, bleu, and tool function-call metrics. PointwiseMetric and PairwiseMetric are generic model-based metrics that you can customize with your own criteria. Because the service takes the prediction results directly from models as input, the evaluation service can perform both inference and subsequent evaluation on all models supported by Vertex AI.

Airflow provides [RunEvaluationOperator](https://github.com/apache/airflow/blob/d5467d6818ce7f54abd1a7a84c30f321f63405c5/airflow/providers/google/cloud/operators/vertex_ai/generative_model.py#L741) to interact with the Vertex AI Rapid Evaluation API.

Learn more: [Google Cloud Vertex AI Evaluation API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/evaluation)



