import unittest
from unittest.mock import patch, MagicMock
import logging
import time
from io import StringIO
from s3helper import S3Helper, S3HelperAutoConfig, S3HelperAutoTokenizer, S3HelperAutoModelForCausalLM, s3_load_dataset
import os
from unittest.mock import ANY
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
        cls.model_name = "jan-hq-test/tokenizer-tinyllama"
        cls.dataset_name = "jan-hq-test/test-dataset"
        S3Helper()

    @test_name("Tokenizer Loading Test")
    @patch('s3helper.S3HelperAutoTokenizer.from_pretrained')
    def test_tokenizer_loading(self, mock_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = S3HelperAutoTokenizer.from_pretrained(self.model_name)
        
        mock_from_pretrained.assert_called_once_with(self.model_name)
        self.assertIsNotNone(tokenizer)
        self.assertEqual(tokenizer, mock_tokenizer)

    @test_name("Dataset Loading Test")
    # @patch('s3helper.s3_load_dataset')
    def test_dataset_loading(self):
        try:
            dataset = s3_load_dataset(self.dataset_name, file_format='parquet', split='train')
            print(dataset)
            self.assertIsNotNone(dataset)
        except Exception as e:
            self.fail(f"s3_load_dataset raised an exception: {e}")
    @test_name("Config Loading Test")
    @patch('s3helper.S3HelperAutoConfig.from_pretrained')
    def test_config_loading(self, mock_from_pretrained):
        mock_config = MagicMock()
        mock_from_pretrained.return_value = mock_config

        config = S3HelperAutoConfig.from_pretrained(self.model_name)
        
        mock_from_pretrained.assert_called_once_with(self.model_name)
        self.assertIsNotNone(config)
        self.assertEqual(config, mock_config)

    @test_name("Model Loading Test")
    @patch('s3helper.S3HelperAutoModelForCausalLM.from_pretrained')
    def test_model_loading(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = S3HelperAutoModelForCausalLM.from_pretrained(self.model_name)
        
        mock_from_pretrained.assert_called_once_with(self.model_name)
        self.assertIsNotNone(model)
        self.assertEqual(model, mock_model)
if __name__ == '__main__':
    runner = CustomTestRunner()
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestS3Helper)
    result = runner.run(test_suite)
    runner.print_results()