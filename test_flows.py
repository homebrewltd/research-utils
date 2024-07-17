from s3helper import S3Helper,S3HelperAutoConfig,S3HelperAutoTokenizer,S3HelperAutoModelForCausalLM, s3_load_dataset
import os
import logging

# os.environ['S3_ACCESS_KEY'] = 'minioadmin'
# os.environ['S3_SECRET_KEY'] = 'minioadmin'
# os.environ['S3_ENDPOINT_URL'] = 'http://172.17.0.2:9000'
S3Helper()

# # Example usage
model_name = "jan-hq-test/tokenizer-tinyllama"
# model = S3HelperAutoModelForCausalLM.from_pretrained(model_name)
tokenizer = S3HelperAutoTokenizer.from_pretrained(model_name)
logging.info(f"Tokenizer Loading successful: {tokenizer}")
# print(tokenizer)
# config = S3HelperAutoConfig.from_pretrained(model_name)
# Make sure S3Helper is initialized and environment variables are set
# Load a dataset from S3 bucket
dataset = s3_load_dataset("jan-hq-test/test-dataset",file_format='parquet', split='train')
logging.info(f"Dataset Loading successful")