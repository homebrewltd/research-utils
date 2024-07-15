from s3helper import S3Helper,S3HelperAutoConfig,S3HelperAutoTokenizer,S3HelperAutoModelForCausalLM, s3_load_dataset
import os

os.environ['S3_ACCESS_KEY'] = 'minioadmin'
os.environ['S3_SECRET_KEY'] = 'minioadmin'
os.environ['S3_ENDPOINT_URL'] = 'http://172.17.0.2:9001'
S3Helper()

# # Example usage
# model_name = "thunghiem/tinyllama"
# model = S3HelperAutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = S3HelperAutoTokenizer.from_pretrained(model_name)
# config = S3HelperAutoConfig.from_pretrained(model_name)
# Make sure S3Helper is initialized and environment variables are set
# Load a dataset
dataset = s3_load_dataset("modelhubjan/test_dataset")

# Use the dataset
for item in dataset:
    print(item)

# You can also pass additional arguments to load_dataset
dataset = s3_load_dataset("modelhubjan/test_dataset", file_format='parquet', split='train')