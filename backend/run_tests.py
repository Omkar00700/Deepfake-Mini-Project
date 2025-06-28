
import unittest
import sys
import logging
import os
import coverage
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_report(coverage_data, test_results):
    """Generate a comprehensive test report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        "timestamp": timestamp,
        "summary": {
            "tests_run": test_results.testsRun,
            "failures": len(test_results.failures),
            "errors": len(test_results.errors),
            "skipped": len(test_results.skipped),
            "success_rate": (test_results.testsRun - len(test_results.failures) - len(test_results.errors)) / 
                           max(1, test_results.testsRun) * 100
        },
        "coverage": {
            "total": coverage_data.get_total_covered_count() / max(1, coverage_data.get_total_execable_count()) * 100,
            "files": {}
        },
        "failures": [{"test": test, "message": message} for test, message in test_results.failures],
        "errors": [{"test": test, "message": message} for test, message in test_results.errors]
    }
    
    # Add per-file coverage data
    for file in coverage_data.measured_files():
        file_analysis = coverage_data.analysis2(file)
        if file_analysis:
            total_statements = len(file_analysis[1])
            total_missing = len(file_analysis[2])
            covered = total_statements - total_missing
            report["coverage"]["files"][file] = {
                "statements": total_statements,
                "missing": total_missing,
                "covered": covered,
                "percentage": (covered / max(1, total_statements)) * 100
            }
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Write report to JSON file
    with open('reports/test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also generate CSV version of report
    generate_csv_report(report, 'reports/test_report.csv')
    
    # Generate summary report file for quick reference
    generate_summary_report(report, 'reports/test_summary.txt')
    
    return report

def generate_csv_report(report_data, output_file):
    """Generate CSV version of the test report"""
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("Test Results Summary\n")
        f.write("=====================\n")
        f.write(f"Timestamp,Tests Run,Failures,Errors,Skipped,Success Rate\n")
        f.write(f"{report_data['timestamp']},{report_data['summary']['tests_run']},"
                f"{report_data['summary']['failures']},{report_data['summary']['errors']},"
                f"{report_data['summary']['skipped']},{report_data['summary']['success_rate']:.2f}%\n\n")
        
        # Write coverage section
        f.write("Coverage Summary\n")
        f.write("================\n")
        f.write(f"Total Coverage,{report_data['coverage']['total']:.2f}%\n\n")
        
        # Write file coverage
        f.write("File Coverage\n")
        f.write("============\n")
        f.write("File,Statements,Missing,Covered,Percentage\n")
        
        for file, data in report_data['coverage']['files'].items():
            f.write(f"{file},{data['statements']},{data['missing']},{data['covered']},{data['percentage']:.2f}%\n")
        
        # Write failures and errors
        if report_data['failures']:
            f.write("\nTest Failures\n")
            f.write("============\n")
            for failure in report_data['failures']:
                f.write(f"{failure['test']},\"{failure['message'].replace('\"', '\"\"')}\"\n")
        
        if report_data['errors']:
            f.write("\nTest Errors\n")
            f.write("==========\n")
            for error in report_data['errors']:
                f.write(f"{error['test']},\"{error['message'].replace('\"', '\"\"')}\"\n")

def generate_summary_report(report_data, output_file):
    """Generate a human-readable summary report"""
    
    with open(output_file, 'w') as f:
        f.write("DeepDefend Test Summary Report\n")
        f.write("=============================\n\n")
        f.write(f"Generated: {report_data['timestamp']}\n\n")
        
        f.write("Test Results\n")
        f.write("-----------\n")
        f.write(f"Tests Run: {report_data['summary']['tests_run']}\n")
        f.write(f"Passed: {report_data['summary']['tests_run'] - report_data['summary']['failures'] - report_data['summary']['errors']}\n")
        f.write(f"Failures: {report_data['summary']['failures']}\n")
        f.write(f"Errors: {report_data['summary']['errors']}\n")
        f.write(f"Skipped: {report_data['summary']['skipped']}\n")
        f.write(f"Success Rate: {report_data['summary']['success_rate']:.2f}%\n\n")
        
        f.write("Coverage Summary\n")
        f.write("---------------\n")
        f.write(f"Total Coverage: {report_data['coverage']['total']:.2f}%\n\n")
        
        # Top 5 files with lowest coverage
        if report_data['coverage']['files']:
            sorted_files = sorted(
                [(file, data['percentage']) for file, data in report_data['coverage']['files'].items()],
                key=lambda x: x[1]
            )
            
            f.write("Files Needing Attention (Lowest Coverage)\n")
            f.write("----------------------------------------\n")
            for file, percentage in sorted_files[:5]:
                f.write(f"{file}: {percentage:.2f}%\n")
            f.write("\n")
        
        # Summary of failures and errors
        if report_data['failures'] or report_data['errors']:
            f.write("Failed Tests Summary\n")
            f.write("-------------------\n")
            for failure in report_data['failures']:
                test_name = str(failure['test']).split()[0]
                f.write(f"FAIL: {test_name}\n")
            
            for error in report_data['errors']:
                test_name = str(error['test']).split()[0]
                f.write(f"ERROR: {test_name}\n")
        else:
            f.write("All tests passed successfully!\n")

if __name__ == '__main__':
    # Whether to run with coverage
    run_with_coverage = '--coverage' in sys.argv
    
    if run_with_coverage:
        # Start code coverage
        logger.info("Running tests with code coverage")
        cov = coverage.Coverage(
            source=[
                'preprocessing.py', 
                'inference.py',
                'postprocessing.py',
                'detection_handler.py',
                'model_manager.py',
                'metrics.py',
                'face_detector.py',
                'model.py',
                'debug_utils.py'  # Added debug_utils to coverage
            ],
            omit=['*test*', '*__pycache__*']
        )
        cov.start()
    
    logger.info("Running DeepDefend unit tests")
    
    # Discover and run tests
    test_suite = unittest.defaultTestLoader.discover('.', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Process test results
    if run_with_coverage:
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        
        logger.info("Generating code coverage report")
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Generate console report
        cov.report()
        
        # Generate HTML report
        cov.html_report(directory='reports/coverage_html')
        
        # Generate XML report for CI integration
        cov.xml_report(outfile='reports/coverage.xml')
        
        # Generate comprehensive test report
        test_report = generate_test_report(cov, result)
        
        logger.info(f"Test Summary: {test_report['summary']['tests_run']} tests, "
                   f"{test_report['summary']['failures']} failures, "
                   f"{test_report['summary']['errors']} errors, "
                   f"Success Rate: {test_report['summary']['success_rate']:.1f}%")
        
        logger.info(f"Coverage: {test_report['coverage']['total']:.1f}%")
        logger.info("Code coverage report generated in reports/coverage_html")
        logger.info("Test reports generated in reports/ directory")
    
    # Return appropriate exit code
    sys.exit(0 if result.wasSuccessful() else 1)
