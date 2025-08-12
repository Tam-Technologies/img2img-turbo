#!/usr/bin/env python3

import argparse
import os
import subprocess

from src import constants

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

IMAGE_CHOICES = ['train', 'inference']

def build(args):
    print(f"Building Docker image")
    dockerfile_path = os.path.join(SCRIPT_DIR, f"{args.image}_app", "Dockerfile")

    docker_build_cmd = ["docker", "build",  "--tag", f"{constants.IMAGE_NAME}-{args.image}", "-f", dockerfile_path]
    if args.image == 'train':
        docker_build_cmd.extend(["--build-arg", "WANDB_API_KEY=" + os.environ['WANDB_API_KEY']])
    docker_build_cmd.append(SCRIPT_DIR)
    print(' '.join(docker_build_cmd))
    subprocess.check_call(docker_build_cmd)

def push(args):
    build(args)

    print(f"Pushing Docker image {constants.IMAGE_NAME}-{args.image} to Google Container Registry")
    registry_image = f"gcr.io/{constants.PROJECT_ID}/{constants.IMAGE_NAME}-{args.image}"
    subprocess.check_call(["docker", "image", "tag", f"{constants.IMAGE_NAME}-{args.image}", registry_image])
    subprocess.check_call(["docker", "push", registry_image])

def deploy(args):
    push(args)

    if args.image == "train":
        print(f"{args.image} is not a Google Cloud Run service, and should deployed to Vertex AI instead. Exiting.")
        return

    # In order to set up a Google Cloud Storage bucket as a volume that is mounted to the Docker container, follow the
    # instructions here:
    # https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts#console
    # This only needs to be done once. Once mounted, the container can read/write directly to the bucket.
    registry_image = f"gcr.io/{constants.PROJECT_ID}/{constants.IMAGE_NAME}-{args.image}"
    print(f"Deploying Docker service {constants.IMAGE_NAME}-{args.image} to Google Run")
    memory = "16Gi"
    cpu = "4"
    gpu = "1"
    gpu_type = "nvidia-l4"
    max_instances = "3"
    deploy_cmd = ["gcloud", "run", "deploy", f"{constants.IMAGE_NAME}-{args.image}", "--image", registry_image,
                  "--project", constants.PROJECT_ID, "--region", "us-central1", "--memory", memory, "--cpu", cpu,
                  "--gpu", gpu, "--gpu-type", gpu_type, "--max-instances", max_instances, "--timeout", "30m"]

    print(' '.join(deploy_cmd))
    subprocess.check_call(deploy_cmd)

def run(args):
    build(args)

    PORT=8080
    image_key_path="/tmp/keys/google_key.json"
    local_key_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    docker_run_cmd = ["docker", "run"]
    docker_run_opt = ["-p", f"8080:{PORT}", "-e", f"PORT={PORT}", "-e", "K_SERVICE=dev",
                           "-e", "K_CONFIGURATION=dev", "-e", "K_REVISION=dev-00001",
                           "-e", f"GOOGLE_APPLICATION_CREDENTIALS={image_key_path}",
                           "-v", f"{local_key_path}:{image_key_path}:ro",
                           f"{constants.IMAGE_NAME}-{args.image}"]
    if args.shell:
        docker_run_cmd.append("-it")
        docker_run_opt.append("sh")
    docker_run_cmd.extend(docker_run_opt)
    print(' '.join(docker_run_cmd))
    subprocess.check_call(docker_run_cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image utilities')
    parser.add_argument('-i', '--image', required=True, choices=IMAGE_CHOICES, help='Name of the image')
    subparsers = parser.add_subparsers(help='Choose what you would like to do with this image. Default is build')
    parser.set_defaults(func=run)
    parser.set_defaults(shell=False)

    parser_build = subparsers.add_parser('build', help='Build the image locally')
    parser_build.set_defaults(func=build)

    parser_build = subparsers.add_parser('push', help='Push the image to Google Container Registry')
    parser_build.set_defaults(func=push)

    parser_deploy = subparsers.add_parser('deploy', help='Deploy the image to Google Cloud Run')
    parser_deploy.set_defaults(func=deploy)

    parser_run = subparsers.add_parser('run', help='Run the image locally')
    parser_run.set_defaults(func=run)
    parser_run.add_argument("-sh", "--shell", action="store_true", help="Run the image locally in an interactive shell")

    args = parser.parse_args()
    args.func(args)