import os
import boto3
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import sys
import logging
from datasets import load_dataset, Dataset, load_from_disk
from typing import Optional, Dict, Any, List
# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')

def find_files(directory: str, file_format: str):
    matching_files = []
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file ends with the specified format
            if file.endswith(f".{file_format}"):
                matching_files.append(os.path.join(root, file))
    
    return matching_files

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

    def download_file(self, path_components: list, local_dir: str):
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
    
    def ensure_file_local(self, file_name_or_path: str, local_dir: str):
        path_components = file_name_or_path.split("/")
        if len(path_components) != 2:
            logging.error("Cannot recognize bucket name and file name since the components are not 2")
            raise ValueError("Cannot recognize bucket name and file name since the components are not 2")
        file_local_path = os.path.join(local_dir, file_name_or_path)
        if not os.path.exists(file_local_path):
            os.makedirs(file_local_path, exist_ok=True)
            self.download_file(path_components, local_dir)
        else:
            if 'model' in local_dir.lower():
                
                logging.info(f"Model existed at: {file_local_path}, read from cache")
            elif 'dataset' in local_dir.lower():
                logging.info(f"Dataset existed at: {file_local_path}, read from cache")
        return file_local_path

    def upload_to_s3(self, local_dir, bucket_name, file_name):
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_key = os.path.relpath(local_file_path, local_dir)
                self.s3_client.upload_file(local_file_path, bucket_name, os.path.join(file_name, s3_key))
                logging.info(f'Uploaded {local_file_path} to s3://{bucket_name}/{file_name}/{s3_key}')
    # def download_dataset(self, path_components: list, local_dir: str = './datasets'):
    #     bucket_name = path_components[0]
    #     dataset_name = path_components[1]
    #     objects = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=dataset_name)
    #     for obj in objects.get('Contents', []):
    #         file_key = obj['Key']
    #         if file_key.endswith('/'):
    #             continue  # Skip directories
    #         file_path = os.path.join(local_dir, bucket_name, file_key)
    #         os.makedirs(os.path.dirname(file_path), exist_ok=True)
    #         self.s3_client.download_file(bucket_name, file_key, file_path)
    #         logging.info(f'Downloaded dataset file: {file_key}')

class S3HelperAutoModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, local_dir: str = './models', **kwargs):
        s3_helper = S3Helper.get_instance()
        model_local_path = s3_helper.ensure_file_local(pretrained_model_name_or_path, local_dir)
        return super().from_pretrained(model_local_path, *model_args, **kwargs)

class S3HelperAutoTokenizer(AutoTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, local_dir: str = './models', **kwargs):
        s3_helper = S3Helper.get_instance()
        tokenizer_local_path = s3_helper.ensure_file_local(pretrained_model_name_or_path, local_dir)
        return super().from_pretrained(tokenizer_local_path, *model_args, **kwargs)

class S3HelperAutoConfig(AutoConfig):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, local_dir: str = './models', **kwargs):
        s3_helper = S3Helper.get_instance()
        config_local_path = s3_helper.ensure_file_local(pretrained_model_name_or_path, local_dir)
        return super().from_pretrained(config_local_path, *model_args, **kwargs)
# defined a custom load_dataset from S3 bucket
def s3_load_dataset(
    dataset_name_or_path: str,
    file_format: str = 'json',
    local_dir: str = './datasets',
    split: str = None,
    *args: Any,
    **kwargs: Any
) -> Dataset:
    """
    Load a dataset from S3/Minio storage.
    Args:
    dataset_name_or_path (str): Path to the dataset in the format 'bucket_name/dataset_name'
    file_format (str): File format of the dataset. Either 'json', 'csv', or 'parquet'.
    local_dir (str): Local directory to store downloaded datasets
    split (str): Dataset split to load ('train', 'test', or None for all)
    *args: Additional positional arguments to pass to load_dataset
    **kwargs: Additional keyword arguments to pass to load_dataset
    Returns:
    Dataset: The loaded dataset
    """
    s3_helper = S3Helper.get_instance()
    dataset_local_path = s3_helper.ensure_file_local(dataset_name_or_path, local_dir)
    
    def find_files(path: str, extension: str) -> List[str]:
        return [os.path.join(root, file) for root, _, files in os.walk(path) 
                for file in files if file.endswith(f'.{extension}')]
    
    local_files = find_files(dataset_local_path, file_format)
    logging.info(f"Found local files: {local_files}")
    
    data_files: Dict[str, List[str]] = {"train": [], "test": []}
    for file in local_files:
        if "train" in file:
            data_files["train"].append(file)
        elif "test" in file:
            data_files["test"].append(file)
        else:
            logging.warning(f"Unclassified file: {file}")
    
    if split:
        if split not in data_files:
            raise ValueError(f"Invalid split: {split}. Available splits are: {list(data_files.keys())}")
        data_files = {split: data_files[split]}
    
    # Remove empty splits
    data_files = {k: v for k, v in data_files.items() if v}
    
    if not data_files:
        raise ValueError(f"No valid files found for the specified format and split.")
    
    logging.info(f"Loading dataset with data_files: {data_files}")
    return load_dataset(file_format, data_files=data_files, *args, **kwargs)