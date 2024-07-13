import os
import boto3
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import sys

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')

class S3Helper:
    _instance = None

    @staticmethod
    def get_instance():
        if S3Helper._instance is None:
            S3Helper()
        return S3Helper._instance

    def __init__(self):
        if S3Helper._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            endpoint_url = os.getenv('S3_ENDPOINT_URL')
            access_key = os.getenv('S3_ACCESS_KEY')
            secret_key = os.getenv('S3_SECRET_KEY')
            if not access_key or not secret_key:
                raise ValueError("S3 credentials must be set in environment variables S3_ACCESS_KEY and S3_SECRET_KEY")
            
            self.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            self.validate_credentials()
            S3Helper._instance = self

    def validate_credentials(self):
        try:
            self.s3_client.list_buckets()
            logging.info("S3 credentials are valid.")
        except Exception as e:
            logging.error(f"Invalid S3 credentials: {e}")
            raise ValueError("Invalid S3 credentials")

    def download_model(self, path_components: list, local_dir: str = './models'):
        bucket_name = path_components[0]
        model_name = path_components[1]
        objects = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=model_name)
        for obj in objects.get('Contents', []):
            file_key = obj['Key']
            if file_key.endswith('/'):
                continue  # Skip directories
            file_path = os.path.join(local_dir, bucket_name, file_key)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.s3_client.download_file(bucket_name, file_key, file_path)
            logging.info(f'Downloaded file: {file_key}')
    
    def ensure_model_local(self, pretrained_model_name_or_path, local_dir):
        path_components = pretrained_model_name_or_path.split("/")
        if len(path_components) != 2:
            logging.error("Cannot recognize bucket name and model name since having > 2 components")
            raise ValueError("Cannot recognize bucket name and model name since having > 2 components")
        model_local_path = os.path.join(local_dir, pretrained_model_name_or_path)
        if not os.path.exists(model_local_path):
            os.makedirs(model_local_path, exist_ok=True)
            self.download_model(path_components, local_dir)
        else:
            logging.info(f"Model existed at: {model_local_path}, read from cache")
        return model_local_path

    def upload_to_s3(self, local_dir, bucket_name, model_name):
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_key = os.path.relpath(local_file_path, local_dir)
                self.s3_client.upload_file(local_file_path, bucket_name, os.path.join(model_name, s3_key))
                logging.info(f'Uploaded {local_file_path} to s3://{bucket_name}/{model_name}/{s3_key}')

class S3HelperAutoModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, local_dir: str = './models', **kwargs):
        s3_helper = S3Helper.get_instance()
        model_local_path = s3_helper.ensure_model_local(pretrained_model_name_or_path, local_dir)
        return super().from_pretrained(model_local_path, *model_args, **kwargs)

class S3HelperAutoTokenizer(AutoTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, local_dir: str = './models', **kwargs):
        s3_helper = S3Helper.get_instance()
        tokenizer_local_path = s3_helper.ensure_model_local(pretrained_model_name_or_path, local_dir)
        return super().from_pretrained(tokenizer_local_path, *model_args, **kwargs)

class S3HelperAutoConfig(AutoConfig):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, local_dir: str = './models', **kwargs):
        s3_helper = S3Helper.get_instance()
        config_local_path = s3_helper.ensure_model_local(pretrained_model_name_or_path, local_dir)
        return super().from_pretrained(config_local_path, *model_args, **kwargs)
