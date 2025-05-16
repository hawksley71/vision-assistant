import unittest
import sys
import time
from datetime import datetime
import threading
import traceback

class ProgressTestRunner(unittest.TextTestRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        self.total_tests = 0
        self.completed_tests = 0
        self.last_progress_time = None
        self.test_timeout = 30  # Maximum seconds per test
        self._timeout_thread = None
        self.current_test = None

    def run(self, test):
        self.start_time = time.time()
        self.total_tests = test.countTestCases()
        self.completed_tests = 0
        self.last_progress_time = time.time()
        print(f"\nStarting {self.total_tests} tests at {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 50)
        
        # Start watchdog thread
        self._timeout_thread = threading.Thread(target=self._watchdog, daemon=True)
        self._timeout_thread.start()
        
        result = super().run(test)
        
        duration = time.time() - self.start_time
        print("\n" + "=" * 50)
        print(f"Tests completed in {duration:.2f} seconds")
        print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"Failed: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        return result

    def _watchdog(self):
        """Watchdog thread to detect stuck tests."""
        while self.completed_tests < self.total_tests:
            current_time = time.time()
            if current_time - self.last_progress_time > self.test_timeout:
                print(f"\n[WARNING] Test appears to be stuck! No progress for {self.test_timeout} seconds")
                print(f"Last completed test: {self.completed_tests}/{self.total_tests}")
                if self.current_test:
                    print(f"Current test: {self.current_test}")
                    print("Stack trace:")
                    traceback.print_stack()
                print("Current test may be hanging. Consider interrupting and debugging.")
            time.sleep(1)

    def _makeResult(self):
        result = super()._makeResult()
        original_addSuccess = result.addSuccess
        
        def addSuccess(test):
            self.completed_tests += 1
            self.last_progress_time = time.time()
            self.current_test = test.id()
            progress = (self.completed_tests / self.total_tests) * 100
            duration = time.time() - self.start_time
            print(f"\rProgress: {progress:.1f}% ({self.completed_tests}/{self.total_tests}) - {test.id()} - {duration:.2f}s", end="")
            print(f"\n[DEBUG] Test {test.id()} completed in {duration:.2f}s")
            original_addSuccess(test)
        
        result.addSuccess = addSuccess
        return result

if __name__ == '__main__':
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = ProgressTestRunner(verbosity=2)
    runner.run(suite) 