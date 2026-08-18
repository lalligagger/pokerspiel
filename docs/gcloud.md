# Google Cloud Run setup and access notes

This note captures the actual sequence we used to get the FastAPI service reachable and to fix the 403 authorization problem.

## What happened

The app was running locally with FastAPI/Uvicorn on port 8080, and the project was configured to serve the API from Docker Compose:

```yaml
services:
  pokerkit-open-spiel:
    build: .
    working_dir: /app
    volumes:
      - .:/app
    ports:
      - "8080:8080"
    command: uvicorn api.app:app --host 0.0.0.0 --port 8080
```

The local status endpoint worked:

```bash
curl -sS http://127.0.0.1:8080/status
```

This returned a valid status payload, confirming the app itself was healthy and listening correctly.

## The Cloud Run 403 problem

When we tried to access the deployed Cloud Run URL:

```bash
auth: https://pokerspiel-6zsh3y4a7a-uw.a.run.app/status
```

we got:

```text
Forbidden
Your client does not have permission to get URL /status from this server.
```

The key point is that `gcloud auth login` only authenticates the CLI; it does not automatically grant the Cloud Run service public access.

## Root cause

The actual Cloud Run service did not exist under the project/region we were targeting when trying to bind IAM permissions. The failure looked like:

```bash
gcloud run services add-iam-policy-binding pokerspiel --region=us-central1 --member="allUsers" --role="roles/run.invoker"
```

with:

```text
ERROR: (gcloud.run.services.add-iam-policy-binding) NOT_FOUND: Resource 'pokerspiel' of kind 'SERVICE' in region 'us-central1' in project 'pokerspiel' does not exist.
```

This meant:

- auth was fine
- project selection was wrong or the service had not been deployed there
- no Cloud Run service existed at the target name/region yet

## Correct fix

1. Check the active project and service list:

```bash
gcloud config list
gcloud run services list --project <project-id>
```

2. If necessary, switch to the correct project:

```bash
gcloud config set project <correct-project-id>
```

3. Bind public invoker access on the actual deployed service:

```bash
gcloud run services add-iam-policy-binding <service-name> \
  --project=<project-id> \
  --region=<region> \
  --member="allUsers" \
  --role="roles/run.invoker"
```

4. If the service is not deployed yet, deploy it and allow unauthenticated access in the same step:

```bash
gcloud run deploy <service-name> \
  --project=<project-id> \
  --region=<region> \
  --allow-unauthenticated \
  --source .
```

## The actual working path

The service was ultimately reachable after correcting the service/project configuration and ensuring the service existed in the target GCP project. The key operational rule is:

- `gcloud auth login` authenticates the user
- `gcloud run services ...` requires a real Cloud Run service and correct IAM policy
- public access is granted with `roles/run.invoker` for `allUsers`

## Practical lesson

A 403 on a Cloud Run URL does not mean the app is failing. It usually means the platform is blocking the request due to Cloud Run IAM or missing service deployment.

The right mental model is:

- app health is separate from service authorization
- local API health is not the same as public cloud access
- Cloud Run access is controlled by service existence, project, region, and IAM policy

## Recap

We verified:

- local FastAPI app responds on 127.0.0.1:8080/status
- Cloud Run URL returned 403 because the service was not properly authorized or did not exist under the active project/region
- the fix was to ensure the service exists in the right GCP project and grant `roles/run.invoker` to `allUsers` (or deploy with `--allow-unauthenticated`)

This is the setup we used to get the API path aligned with the deploy environment.
