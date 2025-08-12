import json

import firebase_admin
import firebase_admin.firestore
from google.cloud import tasks_v2
from google.protobuf import duration_pb2

from src import constants

SEGMENTATION_SERVICE = 'segmentation'
SEGMENTATION_SERVICE_QUEUE = 'segmentation'
SERVICE_ID = 'tdmvwpqp3q'
SEGMENTATION_SERVICE_URL = f'https://{SEGMENTATION_SERVICE}-{SERVICE_ID}-uc.a.run.app'
TASKS_SERVICE_ACCOUNT_EMAIL = f'tasks-service-account@{constants.PROJECT_ID}.iam.gserviceaccount.com'

# Use the application default credentials to initialize app
cred = firebase_admin.credentials.ApplicationDefault()
try:
    firebase_admin.initialize_app(cred, {'projectId': constants.PROJECT_ID})
except:
    pass

db = firebase_admin.firestore.client()
tasks_client = tasks_v2.CloudTasksClient()

def create_segment_image_task(base64_image, use_32bit=False, preprocess=True, output_image_path=None, deadline=None):
    # Create task triggering segmentation service
    # Construct the fully qualified queue name.
    parent = tasks_client.queue_path(constants.PROJECT_ID, constants.LOCATION, SEGMENTATION_SERVICE_QUEUE)

    # Construct the request body.
    if use_32bit:
        endpoint = '/segment-single-image-32bit'
    else:
        endpoint = '/segment-single-image'

    task = {
        "http_request": {  # Specify the type of request.
            "http_method": tasks_v2.HttpMethod.POST,
            "url": SEGMENTATION_SERVICE_URL + endpoint,  # The full url path that the task will be sent to.
            "oidc_token": {
                "service_account_email": TASKS_SERVICE_ACCOUNT_EMAIL,
                "audience": SEGMENTATION_SERVICE_URL,
            },
            "headers": {"Content-Type": "application/json"}
        }
    }

    payload = {
        "input_image": base64_image,
        "preprocess": preprocess
    }
    if output_image_path is not None:
        payload["output_image_path"] = output_image_path
    converted_payload = json.dumps(payload).encode('utf-8') # request body must be in bytes

    # Add the payload to the request.
    task["http_request"]["body"] = converted_payload

    # Add dispatch deadline for requests sent to the worker.
    if deadline is not None:
        duration = duration_pb2.Duration()
        duration.FromSeconds(deadline)
        task["dispatch_deadline"] = duration

    # Use the client to build and send the task.
    response = tasks_client.create_task(request={"parent": parent, "task": task})

    print("Created task {}".format(response.name))
    return response