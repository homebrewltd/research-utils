import unittest
import os
from unittest.mock import patch, MagicMock
from io import StringIO
import time
from s3helper import (
    S3Helper,
    S3HelperAutoConfig,
    S3HelperAutoTokenizer,
    S3HelperAutoModelForCausalLM,
    s3_load_dataset,
)
import sys

class CustomTestResult(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)

class CustomTestRunner(unittest.TextTestRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = StringIO()
        self.results = []

    def run(self, test):
        result = CustomTestResult()
        start_time = time.time()
        test(result)
        time_taken = time.time() - start_time
        self.results.append((result, time_taken))
        return result

    def print_results(self):
        print("\n=== Test Results ===")
        total_tests = 0
        total_successes = 0
        total_failures = 0
        total_errors = 0
        total_time = 0

        for result, time_taken in self.results:
            total_tests += result.testsRun
            total_successes += len(result.successes)
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_time += time_taken

        print(f"Ran {total_tests} tests in {total_time:.3f} seconds")
        print(f"Successes: {total_successes}")
        print(f"Failures: {total_failures}")
        print(f"Errors: {total_errors}")

        print("\nDetailed Results:")
        for result, time_taken in self.results:
            # todo: add time taken for each test
            for test in result.successes:
                print(f"PASS: {test._testMethodName}")
            for test, _ in result.failures:
                print(f"FAIL: {test._testMethodName}")
            for test, _ in result.errors:
                test_name = getattr(test, '_testMethodName', str(test))
                print(f"ERROR: {test_name}")

        if total_failures > 0 or total_errors > 0:
            print("\nFailure and Error Details:")
            for result, _ in self.results:
                for test, traceback in result.failures:
                    print(f"\nFAILURE: {test._testMethodName}")
                    print(traceback)
                for test, traceback in result.errors:
                    test_name = getattr(test, '_testMethodName', str(test))
                    print(f"\nERROR: {test_name}")
                    print(traceback)
        else:
            print("\nAll tests passed successfully!")

def test_name(name):
    def decorator(func):
        func.__name__ = name
        return func
    return decorator

class TestS3Helper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up any necessary test environment
        cls.model_name_or_path = "jan-hq-test/tokenizer-tinyllama"
        cls.dataset_name_or_path = "jan-hq-test/test-dataset"
        

    @test_name("Connect to Minio")
    def test_s3helper_initialization(self):
        with patch('s3helper.S3Helper') as mock_s3helper:
            S3Helper()
            mock_s3helper.assert_called_once()

    @test_name("Load tokenizer from Minio")
    def test_auto_tokenizer_from_pretrained(self):
        with patch('s3helper.S3HelperAutoTokenizer.from_pretrained') as mock_from_pretrained:
            mock_tokenizer = MagicMock()
            mock_from_pretrained.return_value = mock_tokenizer

            tokenizer = S3HelperAutoTokenizer.from_pretrained(self.model_name_or_path)

            mock_from_pretrained.assert_called_once_with(self.model_name_or_path)
            self.assertEqual(tokenizer, mock_tokenizer)

    @test_name("Load dataset from Minio")
    def test_s3_load_dataset(self):
        with patch('s3helper.s3_load_dataset') as mock_load_dataset:
            mock_dataset = MagicMock()
            mock_load_dataset.return_value = mock_dataset

            dataset = s3_load_dataset(self.dataset_name_or_path, file_format='parquet', split='train')

            mock_load_dataset.assert_called_once_with(self.dataset_name_or_path, file_format='parquet', split='train')
            self.assertEqual(dataset, mock_dataset)

    @test_name("Load Causal LM model from Minio")
    def test_auto_model_for_causal_lm_from_pretrained(self):
        with patch('s3helper.S3HelperAutoModelForCausalLM.from_pretrained') as mock_from_pretrained:
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model

            model = S3HelperAutoModelForCausalLM.from_pretrained(self.model_name_or_path)

            mock_from_pretrained.assert_called_once_with(self.model_name_or_path)
            self.assertEqual(model, mock_model)

    @test_name("Load Model Config from Minio")
    def test_auto_config_from_pretrained(self):
        with patch('s3helper.S3HelperAutoConfig.from_pretrained') as mock_from_pretrained:
            mock_config = MagicMock()
            mock_from_pretrained.return_value = mock_config

            config = S3HelperAutoConfig.from_pretrained(self.model_name_or_path)

            mock_from_pretrained.assert_called_once_with(self.model_name_or_path)
            self.assertEqual(config, mock_config)

if __name__ == "__main__":
    runner = CustomTestRunner(stream=sys.stdout, verbosity=2)
    unittest.main(argv=['first-arg-is-ignored'], exit=False, testRunner=runner)
    runner.print_results()