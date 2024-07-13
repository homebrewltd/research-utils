from s3helper import S3Helper,S3HelperAutoConfig,S3HelperAutoTokenizer,S3HelperAutoModelForCausalLM

S3Helper()

# Example usage
model_name = "thunghiem/tinyllama"
model = S3HelperAutoModelForCausalLM.from_pretrained(model_name)
tokenizer = S3HelperAutoTokenizer.from_pretrained(model_name)
config = S3HelperAutoConfig.from_pretrained(model_name)
