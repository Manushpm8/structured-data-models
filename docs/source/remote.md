---
orphan: true
---

# Remote Inference

Use NVIDIA NIM to run a structured-data model on a GPU host and call it from Python with `structured-data-models-client`.
The client prepares requests and returns predictions without downloading model weights or requiring a local GPU.

| Workflow                           | Python package                                 | Input and output                                                         |
| ---------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------ |
| Run a model in your Python process | `structured-data-models` (`sdm`)               | Tensor containers and model predictions; see [In-Context Learning](icl). |
| Call a deployed inference service  | `structured-data-models-client` (`sdm_client`) | DataFrames or predictive queries, a `Request`, and a `PredictionResult`. |

`sdm.models.KumoTabular` and `sdm.models.KumoRelational` are model implementations.
The client's `KumoTabular()` and `KumoRelational()` are handles that select a model on the endpoint.
They share names, but belong to different packages and have different interfaces.

## Prerequisites

You need a reachable inference service and a compatible client installation.
The examples below use the **1.0.0 client API** and assume access to a matching client wheel bundle; do not assume that this package is available on public PyPI.
The bundle must include the client and its matching `structured-data-models-api` and `structured-data-models-pql` wheels, all at version 1.0.0.
Install from the directory containing those wheels:

```bash
python -m pip install --find-links /path/to/client-wheels \
    "structured-data-models-client==1.0.0"
```

Kumo Relational also needs the matching 1.0.0 `structured-data-models-relational-engine` and `structured-data-models-connectors` wheels:

```bash
python -m pip install --find-links /path/to/client-wheels \
    "structured-data-models-client[relational]==1.0.0"
```

Third-party dependencies are resolved from your configured package index.
The base client supports Python 3.10 through 3.14.
The relational-engine wheels are built for CPython 3.10 through 3.13 on Linux with glibc 2.28 or newer, on x86-64 and aarch64.
The tabular client can run on macOS or Windows; installing the relational-engine wheels needs Linux.
The inference service owns the GPU in both cases.

## Connect to a NIM

Choose a model and follow its deployment instructions on a Linux GPU host with Docker and the NVIDIA Container Toolkit:

| Model ID          | Data                                         | Deployment instructions                                                                                                                                                          |
| ----------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kumo-tabular`    | One table; classification and regression     | [Kumo Tabular on NVIDIA Build](https://build.nvidia.com/nvidia/kumo-tabular)                                                                                                     |
| `kumo-relational` | Related tables and temporal prediction tasks | [Kumo Relational tags on NGC](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/kumo-relational/-/tags), [NVIDIA Build](https://build.nvidia.com/nvidia/kumo-relational) |

Check the deployment pages for available tags, supported hardware, access requirements, and the terms for the selected artifact.
For example, authenticate to the registry with an NGC API key as the password:

```bash
docker login nvcr.io --username '$oauthtoken'
```

Then run the Relational NIM in one terminal, mapping host port 8001 to the container's port 8000:

```bash
docker run --rm --gpus all --shm-size=16GB -p 8001:8000 \
    nvcr.io/nim/nvidia/kumo-relational:1.0.1
```

For the Tabular example, follow its deployment page and expose the service on host port 8000.
The Python examples use `http://localhost:8000` for Tabular and `http://localhost:8001` for Relational.
When the service runs on another machine, replace these addresses with the deployment's reachable base URLs.

Wait for the service to finish starting, then check readiness and the model IDs it advertises:

```python
from sdm_client import SDMClient

with SDMClient(url="http://localhost:8001") as client:
    print(client.health_ready())
    print(client.models())
```

Continue when readiness is `True` and the endpoint advertises the model you intend to use.
Pass the **base URL**, such as `http://localhost:8001`; the client appends the `/v1/...` routes itself.
The model handle selects what to ask that endpoint for; it does not route a request to another deployment.

(remote-authentication)=

### Authentication

Registry authentication grants access to the container image.
Inference authentication is configured separately by the deployment operator.
If an authenticating gateway accepts `X-API-Key`, supply its key explicitly with `SDMClient(url="https://your-nim-host", api_key=...)`.
The client sends that value as `X-API-Key` and requires HTTPS for a credentialed connection to a non-local host.

NVIDIA's hosted API Catalog uses model-specific URLs and Bearer authentication.
The HTTP client in this guide does not implement that hosted transport.
For that connection, use the hosted model's API example on NVIDIA Build.

## Predict with Kumo Tabular

Supply labeled context rows and query rows with the same feature columns.
Leave the target out of the query rows:

```python
import pandas as pd
from sdm_client import KumoTabular, Request, SDMClient

context = pd.DataFrame(
    {
        "age": [31.0, 44.5, 28.0, 51.0, 39.0, 57.0],
        "spend": [120.0, 430.0, 90.0, 510.0, 260.0, 600.0],
        "churn": [0, 1, 0, 1, 0, 1],
    }
)
to_score = pd.DataFrame({"age": [33.0, 47.0], "spend": [150.0, 480.0]})
tabular_request = Request(
    context=context,
    predict=to_score,
    target="churn",
    task="classification",
)

with SDMClient(url="http://localhost:8000") as client:
    result = client.predict(tabular_request, model=KumoTabular())
    print(result.to_pandas())
```

The context teaches the prediction task at inference time, without updating model weights.
Classification returns predictions and class probabilities by default.
For a numerical target, use `task="regression"`.

## Predict with Kumo Relational

Describe the tables, their keys, their relationships, and their time columns.
Then express the prediction with the Predictive Query Language (PQL).
This example asks for each user's spending over the next 30 days:

```python
import pandas as pd
from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
    SDMClient,
)

users = pd.DataFrame({"user_id": range(20)})
orders = pd.DataFrame(
    {
        "order_id": range(240),
        "user_id": [i % 20 for i in range(240)],
        "amount": [10.0 + i % 17 for i in range(240)],
        "ordered_at": pd.date_range("2025-01-01", periods=240, freq="D"),
    }
)
schema = Schema.from_data(
    {"users": users, "orders": orders},
    primary_keys={"users": "user_id", "orders": "order_id"},
    time_columns={"users": None, "orders": "ordered_at"},
    relationships=[Relationship("orders", "user_id", "users", "user_id")],
)
model = KumoRelational()
query = PredictiveQuery(
    "PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id",
    schema,
)
relational_request = query.materialize(
    indices=[1, 2, 3],
    model=model,
    num_context_samples=64,
)

with SDMClient(url="http://localhost:8001") as client:
    result = client.predict(relational_request, model=model)
    print(result.to_pandas())
```

`materialize()` resolves the query, constructs historical labeled examples, and samples the related records needed for prediction.
The model receives an instance table, related tables, and their schema in the request.
The NIM does not connect to your database.

Features come from records at or before the prediction anchor, and labels come from the later target window.
`0, 30, days` denotes `(anchor, anchor + 30 days]`.
Use data spanning comfortably more than the target window so the client can construct historical labels.
The client reports when no eligible context can be built.

### Reuse context with a session

A NIM that supports sessions can retain context across prediction calls for the same task and schema.
Using the schema, model, and query from the Relational example:

```python
more_request = query.materialize(
    indices=[4, 5],
    model=model,
    num_context_samples=64,
)

with SDMClient(url="http://localhost:8001") as client:
    with client.fit(
        relational_request,
        model=model,
        holdout=[1, 2, 3, 4, 5],
    ) as fitted:
        first = fitted.predict(relational_request)
        second = fitted.predict(more_request)
        print(first.to_pandas())
        print(second.to_pandas())
```

`fit()` uploads and pins context; it does not train the model.
The fitted handle sends query data against that context, and its context manager releases the session.
Declare every entity the session may score in `holdout` before pinning context, so those entities cannot become their own labeled examples.
For temporal tasks, this excludes those entities at every anchor; use one-shot prediction when historical examples for the same entity are needed.
If the deployment does not support sessions, use `client.predict()`.
Sessions live in the serving process and can expire, be evicted, or disappear after a restart.
Route session calls to the same serving process; create a new session or use one-shot prediction if the session is gone.

## Read data from Databricks or Snowflake

The source of the data and the host of the model are separate choices.
Warehouse tables can be sampled in place and sent to an external NIM.
Install the matching `structured-data-models-connectors` wheel and the client's `databricks` or `snowflake` extra for the source you use.

The following source examples assume that tables equivalent to `users` and `orders` above already exist in the selected warehouse.
In a Databricks notebook with an existing `spark` session:

```python
from sdm_connectors import DatabricksSource

source = DatabricksSource(spark, catalog="main", schema="sales")
```

This Spark path requires Databricks Runtime 17.3 LTS or serverless environment 4 and later.
For a SQL warehouse connection, use `DatabricksSource(connection, catalog=..., schema=...)` instead.

With an existing Snowpark session or Snowflake connector connection named `snowflake_session`:

```python
from sdm_connectors import SnowflakeSource

source = SnowflakeSource(snowflake_session, database="SALES", schema="PUBLIC")
```

Column names in schema arguments and PQL must match the names reported by the warehouse exactly.
The shared example below assumes the lowercase column names from the DataFrames above.
In Snowflake, those names require columns created with quoted lowercase identifiers.
For ordinary unquoted Snowflake columns, use `USER_ID`, `ORDER_ID`, `ORDERED_AT`, and `AMOUNT` in the schema arguments and PQL, such as `users.USER_ID` and `orders.AMOUNT`.

After choosing one source, build the schema with the same keys and relationships:

```python
from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
    SDMClient,
)

model = KumoRelational()
warehouse_schema = Schema.from_source(
    source,
    {"users": "users", "orders": "orders"},
    primary_keys={"users": "user_id", "orders": "order_id"},
    time_columns={"users": None, "orders": "ordered_at"},
    relationships=[Relationship("orders", "user_id", "users", "user_id")],
)
warehouse_request = PredictiveQuery(
    "PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id",
    warehouse_schema,
).materialize(indices=[1, 2, 3], model=model, num_context_samples=64)

with SDMClient(url="https://your-nim-host") as client:
    print(client.predict(warehouse_request, model=model).to_pandas())
```

Schema construction reads catalog metadata, previews up to 1,000 rows per table, and checks primary keys with a full scan of each eligible table.
Pass `check_keys=False` to skip the key scans when you have already validated those keys.
Materialization samples records with SQL, and the client embeds the sampled data in the inference request.
The source uses your warehouse permissions and leaves connection ownership with you.
The notebook must be able to reach the NIM host over the network; supply its base URL and, if required by the gateway, its API key as described under {ref}`remote-authentication`.

## Call a model hosted on Databricks

Databricks Model Serving uses a platform-specific connection rather than the NIM's HTTP routes.
The endpoint must serve the Kumo Relational MLflow package implementing the `request_json`/`response_json` envelope.
Obtain that model package through your deployment provider, or ask your workspace operator to register it and create the serving endpoint.
This section covers invocation of an existing compatible endpoint.
Install the client's `relational` and `databricks-serving` extras from the same wheel bundle.
Use the **serving endpoint name**, and the Relational request prepared above:

```python
with SDMClient.for_databricks_serving("kumo-relational") as client:
    result = client.predict(relational_request, model=KumoRelational())
    print(result.to_pandas())
```

Authentication comes from the Databricks SDK's configured workspace identity, including the notebook's identity when supported by its environment.
For another workspace or credential, pass `workspace_client=WorkspaceClient(...)` from `databricks.sdk`.
The transport sends one `request_json` row and reads one `response_json` row from the structured-data serving model.
An arbitrary MLflow model with a different input signature is not compatible with that envelope.
This path supports one-shot `predict()`; `fit()` and `models()` are unavailable.
The client checks Databricks' 16 MB limit against the wrapped request; reduce the number of context samples or entities per call when a request exceeds it.
See [Databricks custom-model endpoint documentation](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints) for the platform's invocation interfaces.

## Models hosted on Snowflake

Reading tables with `SnowflakeSource` is supported independently of Snowflake-hosted model inference.
Calling a Kumo Relational model deployed through the Snowflake Model Registry on Snowpark Container Services requires a platform serving transport.
The 1.0.0 client API documented here provides no public Snowflake Model Serving transport.
For a supported client workflow, sample Snowflake tables as above and call a deployed NIM's HTTP endpoint.
See [Snowflake model serving](https://docs.snowflake.com/en/developer-guide/snowflake-ml/inference/real-time-inference-rest-api) for the platform's own serving interfaces.

## Work with prediction results

`client.predict()` and the fitted handle return a `PredictionResult`.
`result.to_pandas()` exposes fields such as `prediction`, `probability_<class>`, and `quantile_<level>`, depending on the task and requested outputs.
Use the returned entity `id` when present, or `row_index` for a positional request, to join predictions to the input.
Do not assume the endpoint preserves row order.

Endpoint capabilities depend on the deployed model and version.
`client.models()` reports the model IDs advertised by a NIM, while `client.capabilities(model)` describes the client's local knowledge and is not a live capability query.
The local SDM model APIs, recipes, and tensor outputs remain documented in [In-Context Learning](icl) and [Data Processing](processing).
